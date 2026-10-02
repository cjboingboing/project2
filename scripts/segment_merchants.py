"""
Assigns each merchant to one of 5 segments, from the category in its tags.

The cleaned `keyword_phrase` is one of 25 fixed categories once case and spacing
are normalised, so segments are a lookup, not a clustering problem.

Input:  data/tables/tbl_merchants.parquet (no transactions needed)
Output: data/tables/processed/merchant_segments

Run with:
    python scripts/segment_merchants.py
"""

from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from process_cleaning import load_merchants

#Paths

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "tables" / "processed"

# category (normalised keyword_phrase) -> segment
SEGMENTS = {
    "Technology & Telecom": [
        "digital goods: books, movies, music",
        "computer programming , data processing, and integrated systems design services",
        "computers, computer peripheral equipment, and software",
        "cable, satellite, and other pay television and radio services",
        "telecom",
    ],
    "Home & Garden": [
        "furniture, home furnishings and equipment shops, and manufacturers, except appliances",
        "florists supplies, nursery stock, and flowers",
        "tent and awning shops",
        "lawn and garden supply outlets, including nurseries",
        "equipment, tool, furniture, and appliance rent al and leasing",
    ],
    "Fashion & Personal Care": [
        "shoe shops",
        "watch, clock, and jewelry repair shops",
        "jewelry, watch, clock, and silverware shops",
        "health and beauty spas",
        "opticians, optical goods, and eyeglasses",
    ],
    "Arts, Media & Stationery": [
        "artist supply and craft shops",
        "gift, card, novelty, and souvenir shops",
        "books, periodicals, and newspapers",
        "stationery, office supplies and printing and writing paper",
        "art dealers and galleries",
        "antique shops - sales, repairs, and restoration services",
    ],
    "Leisure & Transport": [
        "bicycle shops - sales and service",
        "music shops - musical instruments, pianos, and sheet music",
        "hobby, toy and game shops",
        "motor vehicle supplies and new parts",
    ],
}

CATEGORY_TO_SEGMENT = {cat: seg for seg, cats in SEGMENTS.items() for cat in cats}


def add_category(merchants):
    """Normalise keyword_phrase (case and spacing vary) into a fixed category."""
    return merchants.withColumn(
        "category", F.trim(F.regexp_replace(F.lower("keyword_phrase"), r"\s+", " "))
    )


def add_segment(merchants):
    """Look the category up in SEGMENTS; unmapped categories stay null."""
    mapping = F.create_map([F.lit(x) for pair in CATEGORY_TO_SEGMENT.items() for x in pair])
    merchants = merchants.withColumn("segment", mapping[F.col("category")])

    unmapped = merchants.filter(F.col("segment").isNull())
    assert unmapped.count() == 0, (
        "unmapped categories: " + ", ".join(r["category"] for r in unmapped.select("category").distinct().collect())
    )
    return merchants


def report(segments):
    """Print segment sizes and take rate spread."""
    width = 66
    print(f"{' merchant segments ':-^{width}}")
    total = segments.count()
    rows = (
        segments.groupBy("segment")
        .agg(
            F.count("*").alias("merchants"),
            F.countDistinct("category").alias("categories"),
            F.round(F.avg("take_rate"), 2).alias("avg_take"),
            F.round(F.min("take_rate"), 2).alias("min_take"),
            F.round(F.max("take_rate"), 2).alias("max_take"),
        )
        .orderBy(F.desc("merchants"))
        .collect()
    )
    print(f"  {'segment':<26}{'merchants':>10}{'share':>8}{'cats':>6}{'take (min-avg-max)':>22}")
    for r in rows:
        take = f"{r['min_take']}-{r['avg_take']}-{r['max_take']}"
        print(f"  {r['segment']:<26}{r['merchants']:>10,}{r['merchants'] / total:>8.1%}{r['categories']:>6}{take:>22}")
    print(f"  {'TOTAL':<26}{total:>10,}")
    print("=" * width)


def main():
    spark = SparkSession.builder.appName("bnpl_segment_merchants").getOrCreate()

    print("Loading merchants...")
    merchants = load_merchants(spark)

    print("Assigning segments...")
    merchants = add_segment(add_category(merchants))

    segments = merchants.select("merchant_abn", "name", "category", "segment", "revenue_code", "take_rate")
    report(segments)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segments.write.mode("overwrite").parquet(str(OUTPUT_DIR / "merchant_segments"))
    print(f"Written to {OUTPUT_DIR / 'merchant_segments'}")

    print("Done.")


if __name__ == "__main__":
    main()
