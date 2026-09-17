"""
Places every consumer in an SA2 and attaches the SA2-level ABS data.

Consumers only carry a postcode, so each one is assigned the SA2 that holds the
largest share of that postcode's population (see download_external_sa2.py for
how the correspondence is built). `sa2_share` is kept alongside as a confidence
weight: a consumer in a postcode that sits wholly inside one SA2 has a share of
1.0, while a rural postcode spread over a dozen SA2s may be closer to 0.2, and
any feature built on this should be read with that in mind.

Input:  data/tables/processed/abs_poa_sa2_primary, abs_census_sa2
        (run download_external_sa2.py first)
Output: data/tables/processed/consumers_sa2

Run with:
    python scripts/join_external_sa2.py
"""

from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType

#Paths

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "tables"
OUTPUT_DIR = DATA_DIR / "processed"

CONSUMER_PATH = DATA_DIR / "tbl_consumer.csv"
CONSUMER_DETAILS_PATH = DATA_DIR / "consumer_user_details.parquet"
POA_SA2_PATH = OUTPUT_DIR / "abs_poa_sa2_primary"
CENSUS_SA2_PATH = OUTPUT_DIR / "abs_census_sa2"

CONSUMER_SCHEMA = StructType([
    StructField("name", StringType(), True),
    StructField("address", StringType(), True),
    StructField("state", StringType(), True),
    StructField("postcode", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("consumer_id", LongType(), True),
])


def load_consumers(spark):
    """Read tbl_consumer.csv and pad postcodes to 4 digits."""
    consumers = (
        spark.read.schema(CONSUMER_SCHEMA)
        .option("header", "true")
        .option("sep", "|")
        .csv(str(CONSUMER_PATH))
    )
    return consumers.select(
        "consumer_id",
        F.trim("state").alias("state"),
        F.lpad(F.trim("postcode"), 4, "0").alias("postcode"),  # 862 -> 0862
        "gender",
    )


def attach_user_ids(spark, consumers):
    """Add user_id (one-to-one with consumer_id)."""
    details = spark.read.parquet(str(CONSUMER_DETAILS_PATH))
    joined = details.join(consumers, on="consumer_id", how="inner")

    n_details, n_consumers, n_joined = details.count(), consumers.count(), joined.count()
    assert n_details == n_consumers == n_joined, (
        f"details={n_details}, consumers={n_consumers}, joined={n_joined}"
    )
    return joined


def attach_sa2(spark, consumers):
    """Left join the primary SA2 for each postcode, then the SA2-level ABS data."""
    primary = spark.read.parquet(str(POA_SA2_PATH)).select(
        "postcode",
        "sa2_code",
        F.col("sa2_name").alias("sa2_name"),
        F.col("state_name").alias("sa2_state_name"),
        "sa2_share",
        "n_sa2s",
    )
    census = spark.read.parquet(str(CENSUS_SA2_PATH)).drop("sa2_name")

    joined = consumers.join(primary, on="postcode", how="left")
    assert joined.count() == consumers.count(), "postcode join duplicated consumers"

    joined = joined.withColumn("sa2_matched", F.col("sa2_code").isNotNull())
    joined = joined.join(census, on="sa2_code", how="left")
    assert joined.count() == consumers.count(), "SA2 join duplicated consumers"

    abs_cols = [c for c in census.columns if c != "sa2_code"]
    return joined.select(
        "user_id", "consumer_id", "state", "postcode", "gender",
        "sa2_matched", "sa2_code", "sa2_name", "sa2_state_name", "sa2_share", "n_sa2s",
        *abs_cols,
    )


def report_missing(consumers_sa2):
    """Print unmatched consumers, how confident the SA2 assignment is, and nulls."""
    width = 58
    n_total = consumers_sa2.count()
    unmatched = consumers_sa2.filter(~F.col("sa2_matched"))
    n_unmatched = unmatched.count()

    print(f"{' consumer -> SA2 join ':-^{width}}")
    print(f"{'consumers':<38}{n_total:>{width - 38},}")
    print(f"{'placed in an SA2':<38}{n_total - n_unmatched:>{width - 38},}  ({1 - n_unmatched / n_total:.2%})")
    print(f"{'no SA2 for their postcode':<38}{n_unmatched:>{width - 38},}  ({n_unmatched / n_total:.2%})")
    print(f"{'distinct unmatched postcodes':<38}{unmatched.select('postcode').distinct().count():>{width - 38},}")

    if n_unmatched:
        print(f"{' unmatched consumers by state ':-^{width}}")
        for row in unmatched.groupBy("state").count().orderBy(F.desc("count")).collect():
            print(f"  {row['state']:<36}{row['count']:>{width - 38},}")

    print(f"{' how much of the postcode the SA2 covers ':-^{width}}")
    matched = consumers_sa2.filter(F.col("sa2_matched"))
    n_matched = matched.count()
    share = matched.select(
        F.sum((F.col("sa2_share") >= 0.9).cast("int")).alias("strong"),
        F.sum((F.col("sa2_share").between(0.5, 0.9)).cast("int")).alias("fair"),
        F.sum((F.col("sa2_share") < 0.5).cast("int")).alias("weak"),
    ).first()
    for label, count in (("90%+", share["strong"]), ("50-90%", share["fair"]), ("under 50%", share["weak"])):
        print(f"  {label:<36}{count:>{width - 38},}  ({count / n_matched:.1%})")

    print(f"{' nulls per ABS column (all consumers) ':-^{width}}")
    abs_cols = [c for c in consumers_sa2.columns if c.startswith(("abs_", "seifa_"))]
    null_counts = consumers_sa2.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in abs_cols]
    ).first()
    for col in abs_cols:
        print(f"  {col:<46}{null_counts[col]:>{width - 48},}")
    print("=" * width)


def main():
    spark = SparkSession.builder.appName("bnpl_join_external_sa2").getOrCreate()

    if not POA_SA2_PATH.exists():
        raise SystemExit(
            f"{POA_SA2_PATH} not found - run scripts/download_external_sa2.py first."
        )
    for path in (CONSUMER_PATH, CONSUMER_DETAILS_PATH):
        if not path.exists():
            raise SystemExit(f"{path} not found - the Canvas data goes in {DATA_DIR}.")

    print("Loading consumers...")
    consumers = load_consumers(spark)

    print("Attaching user_ids...")
    consumers = attach_user_ids(spark, consumers)

    print("Placing consumers in SA2s...")
    consumers_sa2 = attach_sa2(spark, consumers)
    consumers_sa2.cache()

    report_missing(consumers_sa2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    consumers_sa2.write.mode("overwrite").parquet(str(OUTPUT_DIR / "consumers_sa2"))
    print(f"Written to {OUTPUT_DIR / 'consumers_sa2'}")

    print("Done.")


if __name__ == "__main__":
    main()
