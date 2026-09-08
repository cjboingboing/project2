"""
process.py-- loads in .parqeut and .csv files from date/tables and creates various dataframes 
and tables for later analysis. No outlier detection occurs here, only duplicate handling, impossible
number dropping and column renaming. 

Cleaning of the messy "tags" column in tbl_merchant.parquet was neccessary due to inconsistent "manual"
data entry. fraud probabilities are merged into the transaction dataset and non-recorded isntances are 
nulled for later imputation. 

Cleaned outputs are written to data/tables/processed as .parqet or .csv files dependent on the context.

Run with:
    python process.py

Or if you're on a real spark cluster:
    spark-submit process.py

"""


from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, TimestampType
from pyspark.sql.window import Window
import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

#------------------------ Paths --------------------------------#

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

FOLDERS = [
    "transactions_20210228_20210827_snapshot/", 
    "transactions_20210828_20220227_snapshot/",
    "transactions_20220228_20220828_snapshot/"
]

# ---------------------------------------------------------------------------
# Tunable cleaning parameters — these were eyeballed against a sample during
# development; worth re-checking once this runs over the full dataset.
# ---------------------------------------------------------------------------
ABN_LENGTH = 11
FILLER_WORDS = ["-", "and", "or", "a", "an", "the", "of"]
MIN_KEYWORD_FREQ = 5          # words at/above this count are treated as trustworthy vocabulary
MAX_CORRECTION_DISTANCE = 3   # only auto-apply a Levenshtein correction this close or closer

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

def load_merchants(spark):
    """Read raw merchants and parse the (keywords, revenue_code, take_rate)
    tuple out of the "tags" string."""
    merchants = spark.read.schema(MERCHANT_SCHEMA).parquet(str(MERCHANTS_PATH))

    parts = F.split(F.col("tags"), r"[\)\]]\s*,\s*[\(\[]")
    merchants = (
        merchants
        .withColumn("keyword_phrase", F.trim(F.regexp_replace(parts.getItem(0), r"[\[\(\]\)]", "")))
        .withColumn("revenue_code", F.trim(F.regexp_replace(parts.getItem(1), r"[\[\(\]\)]", "")))
        .withColumn("take_rate_raw", F.trim(F.regexp_replace(parts.getItem(2), r"[\[\(\]\)]", "")))
        .withColumn("take_rate", F.regexp_extract(F.col("take_rate_raw"), r"([\d.]+)", 1).cast("double"))
    )
 
    merchants = merchants.withColumn(
        "keywords",
        F.split(F.lower(F.col("keyword_phrase")), r"\s*,\s*")
    )
 
    # enforce ABN length, null out anything malformed
    merchants = merchants.withColumn(
        "merchant_abn",
        F.when(F.length(F.col("merchant_abn").cast("string")) == ABN_LENGTH, F.col("merchant_abn"))
         .otherwise(F.lit(None))
    )
 
    return merchants

def build_keyword_correction_map(merchants):
    """Derive a {dirty_word: clean_word} correction map: lemmatize plurals,
    then snap rare/misspelled words onto the nearest frequent word by
    Levenshtein distance."""
    word_occurrences = (
        merchants
        .select(F.explode("keywords").alias("keyword"))
        .filter(F.col("keyword") != "")
        .select(F.explode(F.split(F.col("keyword"), r"\s+")).alias("keyword"))
        .filter(F.col("keyword") != "")
        .filter(~F.col("keyword").isin(FILLER_WORDS))
    )
 
    vocab = sorted(r["keyword"] for r in word_occurrences.select("keyword").distinct().collect())
    lemmatizer = WordNetLemmatizer()
    singularised = {w: lemmatizer.lemmatize(w, pos="n") for w in vocab}
 
    keyword_counts_singular = (
        word_occurrences
        .replace(singularised, subset=["keyword"])
        .groupBy("keyword")
        .count()
    )
 
    clean_words_df = (
        keyword_counts_singular
        .filter(F.col("count") >= MIN_KEYWORD_FREQ)
        .select(F.col("keyword").alias("clean_word"))
    )
    dirty_words_df = (
        keyword_counts_singular
        .filter(F.col("count") < MIN_KEYWORD_FREQ)
        .select(F.col("keyword").alias("dirty_word"))
    )
 
    pairs = dirty_words_df.crossJoin(clean_words_df).withColumn(
        "distance", F.levenshtein(F.col("dirty_word"), F.col("clean_word"))
    )
    window = Window.partitionBy("dirty_word").orderBy(F.col("distance").asc())
    nearest_match = (
        pairs
        .withColumn("rank", F.row_number().over(window))
        .filter(F.col("rank") == 1)
        .select("dirty_word", "clean_word", "distance")
    )
 
    mapping_rows = nearest_match.filter(
        (F.col("distance") <= MAX_CORRECTION_DISTANCE) & (~F.col("dirty_word").isin(FILLER_WORDS))
    ).collect()
 
    correction_map = dict(singularised)  # start from the lemmatization map
    correction_map.update({r["dirty_word"]: r["clean_word"] for r in mapping_rows})
    return correction_map

def clean_merchant_keywords(merchants, correction_map):
    """Drop filler words and apply the correction map to every merchant's
    keyword array (native Spark array ops, no UDF)."""
    mapping_expr = F.create_map([F.lit(x) for pair in correction_map.items() for x in pair])
    merchants = merchants.withColumn(
        "keywords_clean",
        F.transform(
            F.filter(F.col("keywords"), lambda w: ~w.isin(FILLER_WORDS)),
            lambda word: F.coalesce(mapping_expr[word], word)
        )
    )
    return merchants

def load_fraud_tables(spark):
    consumer_fraud_p = (
        spark.read.schema(CONSUMER_FRAUD_SCHEMA)
        .option("header", "true")
        .csv(str(CONSUMER_FRAUD_PATH))
        .withColumn("order_datetime", F.col("order_datetime").cast("date"))
    )
 
    merchant_fraud_p = (
        spark.read.schema(MERCHANT_FRAUD_SCHEMA)
        .option("header", "true")
        .csv(str(MERCHANT_FRAUD_PATH))
        .withColumn("order_datetime", F.col("order_datetime").cast("date"))
    )
 
    return consumer_fraud_p, merchant_fraud_p

def attach_fraud_probabilities(transactions, merchant_fraud_p, consumer_fraud_p):
    """Left-join both fraud tables onto transactions; rows with no match
    get null in the new probability column, by construction of a left join."""
    transactions = (
        transactions
        .join(
            merchant_fraud_p.select(
                "merchant_abn", "order_datetime",
                F.col("fraud_probability").alias("merchant_fraud_probability")
            ),
            on=["merchant_abn", "order_datetime"],
            how="left",
        )
        .join(
            consumer_fraud_p.select(
                "user_id", "order_datetime",
                F.col("fraud_probability").alias("consumer_fraud_probability")
            ),
            on=["user_id", "order_datetime"],
            how="left",
        )
    )
    return transactions


def load_transactions(spark_sesh, folders: list = FOLDERS, data_dir=DATA_DIR):
    frames = []
    for i, name in enumerate(folders):
        frame = spark_sesh.read.parquet(str(data_dir / name)).withColumn("_source_order", F.lit(i))
        frames.append(frame)

    transactions = frames[0]
    for frame in frames[1:]:
        transactions = transactions.unionByName(frame)

    return transactions.orderBy("_source_order")

    
def main():
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
 
    spark = SparkSession.builder.appName("bnpl_etl").getOrCreate()
 
    print("Loading and cleaning merchants...")
    merchants = load_merchants(spark)
    correction_map = build_keyword_correction_map(merchants)
    merchants = clean_merchant_keywords(merchants, correction_map)
 
    print("Loading fraud probability tables...")
    consumer_fraud_p, merchant_fraud_p = load_fraud_tables(spark)
 
    print("Loading and unioning transaction snapshots...")
    transactions = load_transactions(folders = FOLDERS, data_dir = DATA_DIR, spark_sesh = spark)
 
    print("Attaching fraud probabilities onto transactions...")
    transactions = attach_fraud_probabilities(transactions, merchant_fraud_p, consumer_fraud_p)
 
    print("Loading consumer details...")
    consumers = spark.read.parquet(str(CONSUMER_DETAILS_PATH))
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
    print(f"Writing cleaned outputs to {OUTPUT_DIR} ...")
    merchants.write.mode("overwrite").parquet(str(OUTPUT_DIR / "merchants_clean"))
    transactions.write.mode("overwrite").parquet(str(OUTPUT_DIR / "transactions_clean"))
    consumer_fraud_p.write.mode("overwrite").parquet(str(OUTPUT_DIR / "consumer_fraud_probability"))
    merchant_fraud_p.write.mode("overwrite").parquet(str(OUTPUT_DIR / "merchant_fraud_probability"))
    consumers.write.mode("overwrite").parquet(str(OUTPUT_DIR / "consumers"))
 
    print("Done.")

if __name__ == "__main__":
    main()
