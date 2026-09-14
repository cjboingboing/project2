"""
Downloads ABS 2021 Census data by Postal Area (POA) and writes a
small postcode-level table describing the area each consumer lives in. 

Source: ABS 2021 Census General Community Profile (GCP) DataPack, Postal Areas, short headers
    G01 -- Selected person characteristics  -> total population
    G02 -- Selected medians and averages    -> median age, income, rent, mortgage repayment

Decisions made:
    - Postal Areas rather than SA2. Consumers only have a postcode and the ABS publishes no
      official postcode.
    - A median of 0 is treated as missing (null).
    - Use 2021 Census data since it's the closest Census to the transaction dates.

Outputs:
    data/tables/external/   raw zip + the two extracted CSVs (gitignored)
    data/tables/processed/abs_census_poa   one row per postcode, parquet

Run with:
    python scripts/download_external.py
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

#------------------------ Paths --------------------------------#

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

# ---------------------------------------------------------------------------
# Columns kept, raw ABS short header -> our name. Meanings checked against
# Metadata/Metadata_2021_GCP_DataPack_R1_R2.xlsx inside the zip.
# ---------------------------------------------------------------------------
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
    """Download the DataPack once. If it is already on disk, skip, so reruns
    don't hit the ABS site again."""
    if dest.exists():
        print(f"Already downloaded, skipping: {dest.name}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f)
    # only give it the real name once complete, so a crashed download isn't
    # mistaken for a finished one on the next run
    partial.rename(dest)
    return dest


def extract_tables(zip_path, out_dir=EXTERNAL_DIR):
    """Pull just G01 and G02 out of the zip; the DataPack holds many tables we don't use."""
    paths = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in (G01_CSV, G02_CSV):
            target = out_dir / name
            with z.open(ZIP_TABLE_DIR + name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            paths[name] = target
    return paths


def build_census_table(spark, g01_path, g02_path):
    """Keep the chosen columns, join G01 and G02 on postal area, turn 'POA3000'
    into the postcode '3000', and null out zero medians."""
    g01 = spark.read.option("header", "true").csv(str(g01_path))
    g02 = spark.read.option("header", "true").csv(str(g02_path))

    g01 = g01.select("POA_CODE_2021", *[F.col(raw).cast("int").alias(new) for raw, new in G01_COLUMNS.items()])
    g02 = g02.select("POA_CODE_2021", *[F.col(raw).cast("int").alias(new) for raw, new in G02_COLUMNS.items()])

    census = g01.join(g02, on="POA_CODE_2021", how="inner")

    # G01 and G02 cover the same postal areas, so the join shouldn't lose any
    n_g01, n_g02, n_joined = g01.count(), g02.count(), census.count()
    assert n_g01 == n_g02 == n_joined, f"G01={n_g01}, G02={n_g02}, joined={n_joined}"

    # kept as a string so NT postcodes like 0800 keep their leading zero
    census = census.withColumn("postcode", F.regexp_extract("POA_CODE_2021", r"^POA(\d{4})$", 1))

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
