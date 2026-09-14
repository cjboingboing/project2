"""
Joins ABS Census data onto consumers by postcode, one row per user_id.

Input:  data/tables/processed/abs_census_poa (run download_external.py first)
Output: data/tables/processed/consumers_abs

Run with:
    python scripts/join_external.py
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
ABS_CENSUS_PATH = OUTPUT_DIR / "abs_census_poa"

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


def attach_census(spark, consumers):
    """Left join ABS on postcode; abs_matched = False where no postal area exists."""
    census = spark.read.parquet(str(ABS_CENSUS_PATH))
    census = census.withColumn("abs_matched", F.lit(True))

    joined = consumers.join(census, on="postcode", how="left")
    joined = joined.withColumn("abs_matched", F.coalesce(F.col("abs_matched"), F.lit(False)))

    assert joined.count() == consumers.count(), "postcode join duplicated consumers"

    abs_cols = [c for c in census.columns if c.startswith("abs_") and c != "abs_matched"]
    return joined.select("user_id", "consumer_id", "state", "postcode", "gender", "abs_matched", *abs_cols)


def report_missing(consumers_abs):
    """Print unmatched consumers and nulls per ABS column."""
    width = 50
    n_total = consumers_abs.count()
    unmatched = consumers_abs.filter(~F.col("abs_matched"))
    n_unmatched = unmatched.count()

    print(f"{' ABS join ':-^{width}}")
    print(f"{'consumers':<30}{n_total:>{width - 30},}")
    print(f"{'no ABS postal area':<30}{n_unmatched:>{width - 30},}  ({n_unmatched / n_total:.2%})")
    print(f"{'distinct unmatched postcodes':<30}{unmatched.select('postcode').distinct().count():>{width - 30},}")

    print(f"{' unmatched consumers by state ':-^{width}}")
    for row in unmatched.groupBy("state").count().orderBy(F.desc("count")).collect():
        print(f"  {row['state']:<28}{row['count']:>{width - 30},}")

    print(f"{' nulls per ABS column (all consumers) ':-^{width}}")
    abs_cols = [c for c in consumers_abs.columns if c.startswith("abs_") and c != "abs_matched"]
    null_counts = consumers_abs.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in abs_cols]
    ).first()
    for col in abs_cols:
        print(f"  {col:<40}{null_counts[col]:>{width - 42},}")
    print("=" * width)


def main():
    spark = SparkSession.builder.appName("bnpl_join_external").getOrCreate()

    print("Loading consumers...")
    consumers = load_consumers(spark)

    print("Attaching user_ids...")
    consumers = attach_user_ids(spark, consumers)

    print("Joining ABS Census postal areas...")
    consumers_abs = attach_census(spark, consumers)

    report_missing(consumers_abs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    consumers_abs.write.mode("overwrite").parquet(str(OUTPUT_DIR / "consumers_abs"))
    print(f"Written to {OUTPUT_DIR / 'consumers_abs'}")

    print("Done.")


if __name__ == "__main__":
    main()
