"""
Assignment 2 — raw-data cleaning and standardisation only.

This script intentionally DOES NOT perform:
- outlier detection/removal
- log transformations
- date feature engineering
- statistical imputation
- exploratory summaries/plots

Those steps belong in the EDA/imputation notebook.

Outputs
-------
data/tables/processed/
    merchants_clean/
    transactions_clean/
    consumers/
    consumer_fraud_probability/
    merchant_fraud_probability/
    duplicate_order_ids/

Important duplicate policy
--------------------------
`order_id` is treated as the transaction identifier.

For the transaction table:
1. rows missing essential transaction fields are removed;
2. all rows sharing a duplicated order_id are saved to duplicate_order_ids/;
3. one row per order_id is kept in transactions_clean/, via Spark's
   dropDuplicates(["order_id"]).

Which row survives step 3 is whatever Spark's dropDuplicates picks, not a
chosen "best" row: order_datetime has no real time-of-day (the raw
transaction files only carry a partition-level date), so it cannot be used
to break ties meaningfully and is not used to rank duplicate rows.
"""

from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    DoubleType,
    TimestampType,
)
from pyspark.sql.window import Window


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "tables"
OUTPUT_DIR = DATA_DIR / "processed"

MERCHANTS_PATH = DATA_DIR / "tbl_merchants.parquet"
CONSUMER_DETAILS_PATH = DATA_DIR / "consumer_user_details.parquet"
CONSUMER_FRAUD_PATH = DATA_DIR / "consumer_fraud_probability.csv"
MERCHANT_FRAUD_PATH = DATA_DIR / "merchant_fraud_probability.csv"

TRANSACTION_SNAPSHOT_DIRS = [
    DATA_DIR / "transactions_20210228_20210827_snapshot",
    DATA_DIR / "transactions_20210828_20220227_snapshot",
    DATA_DIR / "transactions_20220228_20220828_snapshot",
]


# ---------------------------------------------------------------------------
# Cleaning configuration
# ---------------------------------------------------------------------------

ABN_LENGTH = 11

FILLER_WORDS = [
    "-",
    "and",
    "or",
    "al",
    "a",
    "an",
    "the",
    "of",
]

MIN_KEYWORD_FREQ = 5
MAX_CORRECTION_DISTANCE = 3

FRAUD_MIN = 0.0
FRAUD_MAX = 100.0

CORE_TRANSACTION_COLUMNS = [
    "order_id",
    "user_id",
    "merchant_abn",
    "order_datetime",
    "dollar_value",
]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

MERCHANT_SCHEMA = StructType([
    StructField("name", StringType(), True),
    StructField("tags", StringType(), True),
    StructField("merchant_abn", LongType(), True),
])

CONSUMER_FRAUD_SCHEMA = StructType([
    StructField("user_id", LongType(), True),
    StructField("order_datetime", TimestampType(), True),
    StructField("fraud_probability", DoubleType(), True),
])

MERCHANT_FRAUD_SCHEMA = StructType([
    StructField("merchant_abn", LongType(), True),
    StructField("order_datetime", TimestampType(), True),
    StructField("fraud_probability", DoubleType(), True),
])


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def require_columns(
    df: DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}. "
            f"Available columns: {df.columns}"
        )


def normalise_scalar_strings(df: DataFrame) -> DataFrame:
    """
    Trim scalar STRING columns and convert blank strings to null.

    ARRAY<STRING> columns such as merchant keywords are handled separately.
    """
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            name = field.name
            value = F.trim(F.col(name))

            df = df.withColumn(
                name,
                F.when(
                    F.col(name).isNull() | (value == ""),
                    F.lit(None).cast("string"),
                ).otherwise(value),
            )

    return df


def assert_unique(
    df: DataFrame,
    keys: list[str],
    table_name: str,
) -> None:
    """
    Check small dimension/lookup tables for conflicting duplicate keys.

    Exact duplicates should already have been removed. A duplicated key here
    therefore means two different records would match the same transaction.
    """
    duplicate = (
        df.groupBy(*keys)
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .collect()
    )

    if duplicate:
        raise ValueError(
            f"{table_name} contains conflicting duplicate key(s) on {keys}. "
            f"Example: {duplicate[0].asDict()}"
        )


# ---------------------------------------------------------------------------
# Merchants
# ---------------------------------------------------------------------------

def load_merchants(spark: SparkSession) -> DataFrame:
    """Load raw merchant data and parse the messy tags field."""
    merchants = (
        spark.read
        .schema(MERCHANT_SCHEMA)
        .parquet(str(MERCHANTS_PATH))
    )

    merchants = normalise_scalar_strings(merchants)

    parts = F.split(
        F.col("tags"),
        r"[\)\]]\s*,\s*[\(\[]",
    )

    merchants = (
        merchants
        .withColumn(
            "keyword_phrase",
            F.trim(
                F.regexp_replace(
                    parts.getItem(0),
                    r"[\[\(\]\)]",
                    "",
                )
            ),
        )
        .withColumn(
            "revenue_code",
            F.trim(
                F.regexp_replace(
                    parts.getItem(1),
                    r"[\[\(\]\)]",
                    "",
                )
            ),
        )
        .withColumn(
            "take_rate_raw",
            F.trim(
                F.regexp_replace(
                    parts.getItem(2),
                    r"[\[\(\]\)]",
                    "",
                )
            ),
        )
        .withColumn(
            "take_rate",
            F.regexp_extract(
                F.col("take_rate_raw"),
                r"([\d.]+)",
                1,
            ).cast("double"),
        )
        .withColumn(
            "keywords",
            F.split(
                F.lower(F.col("keyword_phrase")),
                r"\s*,\s*",
            ),
        )
    )

    # ABN is an identifier. Keep the original numeric representation used by
    # the transaction data, but null malformed values.
    merchants = merchants.withColumn(
        "merchant_abn",
        F.when(
            F.length(
                F.col("merchant_abn").cast("string")
            ) == ABN_LENGTH,
            F.col("merchant_abn"),
        ).otherwise(
            F.lit(None).cast("long")
        ),
    )

    return merchants


def split_keyword_phrase(phrase):
    """Split one comma-separated keyword phrase into punctuation-free words."""
    return F.split(
        F.trim(
            F.regexp_replace(
                F.lower(phrase),
                r"[^a-z0-9\s-]",
                "",
            )
        ),
        r"\s+",
    )


def build_keyword_correction_map(
    merchants: DataFrame,
) -> dict[str, str]:
    """
    Build a small word-level correction map from the merchant vocabulary.

    This follows the logic from notebooks/downloads.ipynb:
    noun lemmatisation + rare-word Levenshtein matching to frequent words.
    """
    words = (
        merchants
        .select(
            F.explode("keywords").alias("keyword")
        )
        .filter(F.col("keyword").isNotNull())
        .filter(F.trim(F.col("keyword")) != "")
        .select(
            F.explode(
                split_keyword_phrase(F.col("keyword"))
            ).alias("keyword")
        )
        .filter(F.col("keyword") != "")
        .filter(
            ~F.col("keyword").isin(FILLER_WORDS)
        )
    )

    vocab = sorted(
        row["keyword"]
        for row in (
            words
            .select("keyword")
            .distinct()
            .collect()
        )
    )

    if not vocab:
        return {}

    lemmatizer = WordNetLemmatizer()

    singularised = {
        word: lemmatizer.lemmatize(
            word,
            pos="n",
        )
        for word in vocab
    }

    counts = (
        words
        .replace(
            singularised,
            subset=["keyword"],
        )
        .groupBy("keyword")
        .count()
    )

    clean_words = (
        counts
        .filter(
            F.col("count") >= MIN_KEYWORD_FREQ
        )
        .select(
            F.col("keyword").alias("clean_word")
        )
    )

    dirty_words = (
        counts
        .filter(
            F.col("count") < MIN_KEYWORD_FREQ
        )
        .select(
            F.col("keyword").alias("dirty_word")
        )
    )

    if clean_words.limit(1).count() == 0:
        return dict(singularised)

    if dirty_words.limit(1).count() == 0:
        return dict(singularised)

    pairs = (
        dirty_words
        .crossJoin(clean_words)
        .withColumn(
            "distance",
            F.levenshtein(
                F.col("dirty_word"),
                F.col("clean_word"),
            ),
        )
    )

    nearest_window = (
        Window
        .partitionBy("dirty_word")
        .orderBy(
            F.col("distance").asc(),
            F.col("clean_word").asc(),
        )
    )

    nearest = (
        pairs
        .withColumn(
            "rank",
            F.row_number().over(nearest_window),
        )
        .filter(F.col("rank") == 1)
        .filter(
            F.col("distance")
            <= MAX_CORRECTION_DISTANCE
        )
        .select(
            "dirty_word",
            "clean_word",
        )
        .collect()
    )

    mapping = dict(singularised)

    mapping.update({
        row["dirty_word"]: row["clean_word"]
        for row in nearest
    })

    return mapping


def clean_merchant_keywords(
    merchants: DataFrame,
    correction_map: dict[str, str],
) -> DataFrame:
    """
    Break the comma-split keyword phrases into a flat list of clean words.

    The correction map is word-level, so each phrase must be split into words
    before fillers are dropped and corrections applied.
    """
    cleaned = F.flatten(
        F.transform(
            F.col("keywords"),
            split_keyword_phrase,
        )
    )

    cleaned = F.filter(
        cleaned,
        lambda word: (
            word.isNotNull()
            & (word != "")
            & (~word.isin(FILLER_WORDS))
        ),
    )

    if correction_map:
        map_items = []

        for dirty, clean in correction_map.items():
            map_items.extend([
                F.lit(dirty),
                F.lit(clean),
            ])

        mapping_expr = F.create_map(*map_items)

        cleaned = F.transform(
            cleaned,
            lambda word: F.coalesce(
                mapping_expr[word],
                word,
            ),
        )

    cleaned = F.array_distinct(cleaned)

    return merchants.withColumn(
        "keywords_clean",
        cleaned,
    )


def finalise_merchants(
    merchants: DataFrame,
) -> DataFrame:
    """
    Perform only cleaning/standardisation.

    Missing take_rate is deliberately left null for the notebook rather than
    statistically imputed here.
    """
    merchants = merchants.dropDuplicates()

    merchants = merchants.dropna(
        subset=["merchant_abn"]
    )

    assert_unique(
        merchants,
        ["merchant_abn"],
        "merchants",
    )

    return merchants


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------

def load_consumers(
    spark: SparkSession,
) -> DataFrame:
    """Load and standardise the consumer lookup."""
    consumers = spark.read.parquet(
        str(CONSUMER_DETAILS_PATH)
    )

    consumers = normalise_scalar_strings(
        consumers
    )

    require_columns(
        consumers,
        ["user_id"],
        "consumers",
    )

    consumers = consumers.withColumn(
        "user_id",
        F.col("user_id").cast("long"),
    )

    if "consumer_id" in consumers.columns:
        consumers = consumers.withColumn(
            "consumer_id",
            F.col("consumer_id").cast("long"),
        )

    if "postcode" in consumers.columns:
        consumers = consumers.withColumn(
            "postcode",
            F.when(
                F.col("postcode").isNull(),
                F.lit(None).cast("string"),
            ).otherwise(
                F.lpad(
                    F.col("postcode").cast("string"),
                    4,
                    "0",
                )
            ),
        )

    if "state" in consumers.columns:
        consumers = consumers.withColumn(
            "state",
            F.upper(
                F.trim(F.col("state"))
            ),
        )

    consumers = (
        consumers
        .dropDuplicates()
        .dropna(subset=["user_id"])
    )

    assert_unique(
        consumers,
        ["user_id"],
        "consumers",
    )

    return consumers


# ---------------------------------------------------------------------------
# Fraud tables
# ---------------------------------------------------------------------------

def clean_fraud_table(
    df: DataFrame,
    entity_key: str,
    table_name: str,
) -> DataFrame:
    """
    Standardise fraud lookup data.

    Fraud scores outside 0-100 are considered invalid and set to null.
    No missing fraud probabilities are statistically imputed.
    """
    require_columns(
        df,
        [
            entity_key,
            "order_datetime",
            "fraud_probability",
        ],
        table_name,
    )

    df = (
        df
        .withColumn(
            "order_datetime",
            F.col("order_datetime").cast("date"),
        )
        .withColumn(
            "fraud_probability",
            F.col("fraud_probability").cast("double"),
        )
        .dropDuplicates()
        .dropna(
            subset=[
                entity_key,
                "order_datetime",
            ]
        )
    )

    invalid = (
        F.col("fraud_probability").isNotNull()
        & (
            (F.col("fraud_probability") < FRAUD_MIN)
            | (F.col("fraud_probability") > FRAUD_MAX)
        )
    )

    df = df.withColumn(
        "fraud_probability",
        F.when(
            invalid,
            F.lit(None).cast("double"),
        ).otherwise(
            F.col("fraud_probability")
        ),
    )

    assert_unique(
        df,
        [
            entity_key,
            "order_datetime",
        ],
        table_name,
    )

    return df


def load_fraud_tables(
    spark: SparkSession,
) -> tuple[DataFrame, DataFrame]:
    consumer_fraud = (
        spark.read
        .schema(CONSUMER_FRAUD_SCHEMA)
        .option("header", "true")
        .csv(str(CONSUMER_FRAUD_PATH))
    )

    merchant_fraud = (
        spark.read
        .schema(MERCHANT_FRAUD_SCHEMA)
        .option("header", "true")
        .csv(str(MERCHANT_FRAUD_PATH))
    )

    consumer_fraud = clean_fraud_table(
        consumer_fraud,
        "user_id",
        "consumer_fraud_probability",
    )

    merchant_fraud = clean_fraud_table(
        merchant_fraud,
        "merchant_abn",
        "merchant_fraud_probability",
    )

    return consumer_fraud, merchant_fraud


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

def load_transactions(
    spark: SparkSession,
) -> DataFrame:
    """
    Load and union transaction snapshots.

    No global orderBy is used; physical row order has no analytical meaning
    and sorting millions of rows would add a costly full shuffle.
    """
    frames = []

    for source_order, path in enumerate(
        TRANSACTION_SNAPSHOT_DIRS
    ):
        frame = (
            spark.read
            .parquet(str(path))
            .withColumn(
                "_source_order",
                F.lit(source_order),
            )
        )

        frames.append(frame)

    transactions = frames[0]

    for frame in frames[1:]:
        transactions = (
            transactions.unionByName(
                frame,
                allowMissingColumns=True,
            )
        )

    return transactions


def clean_transactions(
    transactions: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """
    Standardise transactions and remove duplicate order_ids.

    order_datetime has no real time-of-day (the raw transaction files only
    carry a partition-level date), so it cannot be used to rank duplicate
    rows by "earliest" or "most complete." Duplicates are instead resolved
    with Spark's built-in dropDuplicates; which row survives for a given
    order_id is whatever Spark picks, not a chosen "best" row.

    The post-cast, post-dropna frame is persisted to DISK_ONLY so the cleaned
    output and duplicate-audit output can reuse it without recomputing the
    casts and filter from scratch.
    """
    require_columns(
        transactions,
        CORE_TRANSACTION_COLUMNS,
        "transactions",
    )

    transactions = (
        transactions
        .withColumn(
            "order_id",
            F.trim(
                F.col("order_id").cast("string")
            ),
        )
        .withColumn(
            "user_id",
            F.col("user_id").cast("long"),
        )
        .withColumn(
            "merchant_abn",
            F.col("merchant_abn").cast("long"),
        )
        .withColumn(
            "order_datetime",
            F.col("order_datetime").cast("date"),
        )
        .withColumn(
            "order_date",
            F.col("order_datetime").cast("date"),
        )
        .withColumn(
            "dollar_value",
            F.col("dollar_value").cast("double"),
        )
        .withColumn(
            "order_id",
            F.when(
                F.col("order_id") == "",
                F.lit(None).cast("string"),
            ).otherwise(
                F.col("order_id")
            ),
        )
    )

    # These values identify/describe the actual transaction and should not be
    # fabricated by statistical imputation.
    transactions = (
        transactions
        .dropna(
            subset=CORE_TRANSACTION_COLUMNS
        )
        .persist(StorageLevel.DISK_ONLY)
    )

    duplicate_ids = (
        transactions
        .groupBy("order_id")
        .count()
        .filter(F.col("count") > 1)
        .select("order_id")
    )

    duplicate_order_rows = transactions.join(
        duplicate_ids,
        on="order_id",
        how="inner",
    )

    cleaned = transactions.dropDuplicates(
        ["order_id"]
    )

    return cleaned, duplicate_order_rows


# ---------------------------------------------------------------------------
# Fraud joins
# ---------------------------------------------------------------------------

def attach_fraud_probabilities(
    transactions: DataFrame,
    merchant_fraud: DataFrame,
    consumer_fraud: DataFrame,
) -> DataFrame:
    """
    Left-join fraud scores.

    Fraud scores are reported once per entity per day. Both the fraud
    lookups and transactions carry only a calendar date for order_datetime
    (no real time-of-day exists in either source), so this joins on that
    shared date directly -- there is no finer granularity to preserve.

    A null score means there was no matching fraud record; it does NOT mean
    zero fraud probability.
    """
    merchant_lookup = (
        merchant_fraud
        .select(
            "merchant_abn",
            F.col("order_datetime").alias("order_date"),
            F.col(
                "fraud_probability"
            ).alias(
                "merchant_fraud_probability"
            ),
        )
    )

    consumer_lookup = (
        consumer_fraud
        .select(
            "user_id",
            F.col("order_datetime").alias("order_date"),
            F.col(
                "fraud_probability"
            ).alias(
                "consumer_fraud_probability"
            ),
        )
    )

    transactions = (
        transactions
        .join(
            merchant_lookup,
            on=[
                "merchant_abn",
                "order_date",
            ],
            how="left",
        )
        .join(
            consumer_lookup,
            on=[
                "user_id",
                "order_date",
            ],
            how="left",
        )
    )

    return transactions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    nltk.download(
        "wordnet",
        quiet=True,
    )
    nltk.download(
        "omw-1.4",
        quiet=True,
    )

    # Adaptive execution helps Spark combine/split shuffle partitions based
    # on their actual size. 64 is a saner local starting point than doing
    # hundreds of tiny shuffle tasks for this dataset.
    # Local mode runs driver and executor in one JVM, so without an explicit
    # heap size Spark defaults to 1g -- too small for the shuffles/hash
    # aggregations (fraud-probability joins, order_id dedup) over 14M+ rows.
    spark = (
        SparkSession.builder.config("spark.sql.session.timeZone", "Etc/UTC")
        .appName(
            "assignment2_cleaning_standardisation"
        )
        .config(
            "spark.driver.memory",
            "8g",
        )
        .config(
            "spark.sql.adaptive.enabled",
            "true",
        )
        .config(
            "spark.sql.shuffle.partitions",
            "64",
        )
        .getOrCreate()
    )

    try:
        print(
            "\n"
            + "=" * 72
        )
        print(
            "ASSIGNMENT 2 — CLEANING + STANDARDISATION"
        )
        print(
            "=" * 72
        )

        print(
            "\n1. Loading and cleaning merchants..."
        )

        merchants = load_merchants(
            spark
        )

        correction_map = (
            build_keyword_correction_map(
                merchants
            )
        )

        merchants = clean_merchant_keywords(
            merchants,
            correction_map,
        )

        merchants = finalise_merchants(
            merchants
        )

        print(
            "   Merchant cleaning ready."
        )

        print(
            "\n2. Loading and cleaning consumers..."
        )

        consumers = load_consumers(
            spark
        )

        print(
            "   Consumer cleaning ready."
        )

        print(
            "\n3. Loading and cleaning fraud lookup tables..."
        )

        (
            consumer_fraud,
            merchant_fraud,
        ) = load_fraud_tables(
            spark
        )

        print(
            "   Fraud lookup cleaning ready."
        )

        print(
            "\n4. Loading transaction snapshots..."
        )

        transactions = load_transactions(
            spark
        )

        print(
            "\n5. Standardising and deduplicating transactions..."
        )
        print(
            "   (No duplicate count is forced here; this avoids an "
            "unnecessary full Spark job.)"
        )

        (
            transactions,
            duplicate_order_rows,
        ) = clean_transactions(
            transactions
        )

        print(
            "\n6. Attaching fraud probabilities..."
        )

        transactions = attach_fraud_probabilities(
            transactions,
            merchant_fraud,
            consumer_fraud,
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "\n7. Writing cleaned tables..."
        )

        merchants.write.mode(
            "overwrite"
        ).parquet(
            str(
                OUTPUT_DIR
                / "merchants_clean"
            )
        )
        print(
            "   ✓ merchants_clean"
        )

        consumers.write.mode(
            "overwrite"
        ).parquet(
            str(
                OUTPUT_DIR
                / "consumers"
            )
        )
        print(
            "   ✓ consumers"
        )

        consumer_fraud.write.mode(
            "overwrite"
        ).parquet(
            str(
                OUTPUT_DIR
                / "consumer_fraud_probability"
            )
        )
        print(
            "   ✓ consumer_fraud_probability"
        )

        merchant_fraud.write.mode(
            "overwrite"
        ).parquet(
            str(
                OUTPUT_DIR
                / "merchant_fraud_probability"
            )
        )
        print(
            "   ✓ merchant_fraud_probability"
        )

        # Writing transactions materialises the disk-persisted order_id
        # ranking once.
        transactions.write.mode(
            "overwrite"
        ).parquet(
            str(
                OUTPUT_DIR
                / "transactions_clean"
            )
        )
        print(
            "   ✓ transactions_clean"
        )

        # This now reuses the persisted ranked transaction data rather than
        # rerunning the expensive order_id window.
        duplicate_order_rows.write.mode(
            "overwrite"
        ).parquet(
            str(
                OUTPUT_DIR
                / "duplicate_order_ids"
            )
        )
        print(
            "   ✓ duplicate_order_ids"
        )

        print(
            "\nDone."
        )
        print(
            "Outliers, transformations, missing-value analysis and "
            "statistical imputation are intentionally left for the notebook."
        )

    finally:
        # If the JVM has already been killed by the OS, spark.stop() itself
        # raises another Py4J/ConnectionRefused error. Do not let that
        # secondary error hide the original Spark failure.
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
