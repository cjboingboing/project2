"""
Downloads ABS 2021 Census data (G01, G02) by Postal Area and writes a postcode-level table.

Output: data/tables/processed/abs_census_poa

Run with:
    python scripts/download_external.py
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

#Paths

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "tables"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_DIR = DATA_DIR / "processed"

CENSUS_POA_URL = (
    "https://www.abs.gov.au/census/find-census-data/datapacks/download/"
    "2021_GCP_POA_for_AUS_short-header.zip"
)
CENSUS_POA_ZIP = EXTERNAL_DIR / "2021_GCP_POA_for_AUS_short-header.zip"
ZIP_TABLE_DIR = "2021 Census GCP Postal Areas for AUS/"
G01_CSV = "2021Census_G01_AUST_POA.csv"
G02_CSV = "2021Census_G02_AUST_POA.csv"

# raw ABS column -> our column
G01_COLUMNS = {
    "Tot_P_P": "abs_population",
}
G02_COLUMNS = {
    "Median_age_persons": "abs_median_age",
    "Median_tot_prsnl_inc_weekly": "abs_median_personal_income_weekly",
    "Median_tot_hhd_inc_weekly": "abs_median_household_income_weekly",
    "Median_rent_weekly": "abs_median_rent_weekly",
    "Median_mortgage_repay_monthly": "abs_median_mortgage_repay_monthly",
}


def download_census_zip(url=CENSUS_POA_URL, dest=CENSUS_POA_ZIP):
    """Download the DataPack, skipped if already on disk."""
    if dest.exists():
        print(f"Already downloaded, skipping: {dest.name}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f)
    # rename only once complete
    partial.rename(dest)
    return dest


def extract_tables(zip_path, out_dir=EXTERNAL_DIR):
    """Extract G01 and G02 from the zip."""
    paths = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in (G01_CSV, G02_CSV):
            target = out_dir / name
            with z.open(ZIP_TABLE_DIR + name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            paths[name] = target
    return paths


def build_census_table(spark, g01_path, g02_path):
    """Join G01 and G02 into one row per postcode."""
    g01 = spark.read.option("header", "true").csv(str(g01_path))
    g02 = spark.read.option("header", "true").csv(str(g02_path))

    g01 = g01.select("POA_CODE_2021", *[F.col(raw).cast("int").alias(new) for raw, new in G01_COLUMNS.items()])
    g02 = g02.select("POA_CODE_2021", *[F.col(raw).cast("int").alias(new) for raw, new in G02_COLUMNS.items()])

    census = g01.join(g02, on="POA_CODE_2021", how="inner")

    n_g01, n_g02, n_joined = g01.count(), g02.count(), census.count()
    assert n_g01 == n_g02 == n_joined, f"G01={n_g01}, G02={n_g02}, joined={n_joined}"

    # "POA0800" -> "0800", kept as string for the leading zero
    census = census.withColumn("postcode", F.regexp_extract("POA_CODE_2021", r"^POA(\d{4})$", 1))

    # ABS uses 0 for medians it can't calculate
    for col in G02_COLUMNS.values():
        census = census.withColumn(col, F.when(F.col(col) == 0, F.lit(None)).otherwise(F.col(col)))

    return census.select("postcode", *G01_COLUMNS.values(), *G02_COLUMNS.values())


def main():
    spark = SparkSession.builder.appName("bnpl_external_abs").getOrCreate()

    zip_path = download_census_zip()

    print("Extracting G01 and G02...")
    paths = extract_tables(zip_path)

    print("Building postcode-level Census table...")
    census = build_census_table(spark, paths[G01_CSV], paths[G02_CSV])

    print(f"{census.count():,} postal areas. Medians set to null because ABS reported 0:")
    null_counts = census.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in G02_COLUMNS.values()]).first()
    for col in G02_COLUMNS.values():
        print(f"  {col:<40}{null_counts[col]:>5}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    census.write.mode("overwrite").parquet(str(OUTPUT_DIR / "abs_census_poa"))
    print(f"Written to {OUTPUT_DIR / 'abs_census_poa'}")

    print("Done.")


if __name__ == "__main__":
    main()
