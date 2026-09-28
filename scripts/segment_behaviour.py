"""
Segments merchants by behaviour: k-means on the 6 agreed dimensions.

Dimensions: BNPL revenue, customer reach, engagement, growth, activity, stability.
Skewed dimensions are log-scaled, all are standardised, then k is chosen by
silhouette over k = 3..5 (the spec allows 3-5 segments).

Input:  data/tables/processed/merchant_features (from 01_data_quality_eda.ipynb)
Output: data/tables/processed/merchant_behaviour_segments

Run with:
    python scripts/segment_behaviour.py
"""

from pathlib import Path

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, Window, functions as F

#Paths

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "tables" / "processed"
FEATURES_PATH = OUTPUT_DIR / "merchant_features"

# the 6 dimensions the group agreed on -> column in merchant_features
DIMENSIONS = {
    "commercial_value": "estimated_bnpl_revenue",
    "customer_reach": "customer_penetration",
    "customer_engagement": "transactions_per_customer",
    "growth": "log_value_growth",
    "historical_activity": "active_month_ratio",
    # the group calls this "stability", but value_cv measures the opposite:
    # a high coefficient of variation means erratic monthly takings
    "volatility": "value_cv",
}

# shorter names for the profile table
SHORT = {
    "commercial_value": "revenue",
    "customer_reach": "reach",
    "customer_engagement": "engage",
    "growth": "growth",
    "historical_activity": "active",
    "volatility": "volatil",
}

K_RANGE = [3, 4, 5]
NOTABLE = 0.5   # |z| below this is "mid-range", not worth naming
SEED = 42


def load_features(spark):
    """Read merchant_features, keep the 6 dimensions, drop merchants with no metadata."""
    df = spark.read.parquet(str(FEATURES_PATH))

    for name, col in DIMENSIONS.items():
        df = df.withColumnRenamed(col, name)

    # commercial_value needs take_rate, so it is null exactly for the merchants
    # that trade but have no row in tbl_merchants -- they cannot be segmented
    n_all = df.count()
    df = df.filter(F.col("commercial_value").isNotNull())
    print(f"  {n_all - df.count():,} merchants dropped (no merchant metadata)")

    # genuine gaps: no growth without two comparable windows, no stability under two months
    missing = {name: df.filter(F.col(name).isNull()).count() for name in DIMENSIONS}
    medians = {
        name: df.approxQuantile(name, [0.5], 0.01)[0]
        for name in DIMENSIONS
        if missing[name]
    }
    if medians:
        print("  remaining nulls filled with the median:")
        for name, n in missing.items():
            if n:
                print(f"    {name:<22}{n:>6,}  -> {medians[name]:.4g}")
        df = df.fillna(medians)

    df = df.withColumn("dims_imputed", F.lit(bool(medians)))
    return df


def prepare(df):
    """Percentile-rank every dimension, then standardise.

    Raw dimensions have long right tails; k-means on them puts three quarters of
    merchants in one blob. Ranking spreads merchants evenly and clusters on
    relative position, which is what "a big merchant" means here anyway.
    """
    window = Window.orderBy
    for name in DIMENSIONS:
        df = df.withColumn(name + "_p", F.percent_rank().over(window(F.col(name))))

    cols = [n + "_p" for n in DIMENSIONS]
    df = VectorAssembler(inputCols=cols, outputCol="raw_features").transform(df)
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    return scaler.fit(df).transform(df)


def choose_k(df):
    """Fit k = 3..5 and keep the best silhouette."""
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="segment_id")
    total = df.count()
    best = None
    print("  k   silhouette   largest   sizes")
    for k in K_RANGE:
        model = KMeans(featuresCol="features", predictionCol="segment_id", k=k, seed=SEED).fit(df)
        pred = model.transform(df)
        score = evaluator.evaluate(pred)
        sizes = [r["count"] for r in pred.groupBy("segment_id").count().orderBy("segment_id").collect()]
        print(f"  {k}   {score:>10.3f}   {max(sizes) / total:>6.1%}   {sizes}")
        if best is None or score > best[1]:
            best = (k, score, model)
    print(f"  -> k = {best[0]}")
    return best[2]


def name_segments(profile, dim_names):
    """Label each segment from the dimensions where it differs most from the rest."""
    labels = {}
    for seg, values in profile.items():
        ranked = [d for d in sorted(dim_names, key=lambda d: -abs(values[d])) if abs(values[d]) >= NOTABLE]
        if not ranked:
            labels[seg] = "mid-range on every dimension"
            continue
        parts = [f"{'high' if values[d] > 0 else 'low'} {d.replace('_', ' ')}" for d in ranked[:2]]
        labels[seg] = ", ".join(parts)
    return labels


def report(pred, dim_names):
    """Print each segment's size and its average on every dimension (z-scores)."""
    width = 78
    total = pred.count()

    stats = {
        name: pred.select(F.avg(name).alias("m"), F.stddev(name).alias("s")).first()
        for name in dim_names
    }
    rows = (
        pred.groupBy("segment_id")
        .agg(F.count("*").alias("n"), *[F.avg(n).alias(n) for n in dim_names])
        .orderBy("segment_id")
        .collect()
    )
    profile = {
        r["segment_id"]: {n: (r[n] - stats[n]["m"]) / (stats[n]["s"] or 1) for n in dim_names}
        for r in rows
    }
    labels = name_segments(profile, dim_names)

    print(f"{' behaviour segments ':-^{width}}")
    print(f"  {'id':<4}{'merchants':>10}{'share':>8}  profile (z-score vs all merchants)")
    for r in rows:
        z = "  ".join(f"{SHORT[n]}:{profile[r['segment_id']][n]:+.2f}" for n in dim_names)
        print(f"  {r['segment_id']:<4}{r['n']:>10,}{r['n'] / total:>8.1%}  {z}")
    print()
    for seg, label in sorted(labels.items()):
        print(f"  segment {seg}: {label}")
    print("=" * width)
    return labels


def main():
    spark = SparkSession.builder.appName("bnpl_segment_behaviour").getOrCreate()

    print("Loading merchant features...")
    df = load_features(spark)
    print(f"  {df.count():,} merchants")

    print("Scaling dimensions...")
    df = prepare(df)

    print("Choosing k...")
    model = choose_k(df)
    pred = model.transform(df)

    dim_names = list(DIMENSIONS)
    labels = report(pred, dim_names)

    mapping = F.create_map([F.lit(x) for pair in labels.items() for x in pair])
    out = pred.withColumn("segment", mapping[F.col("segment_id")]).select(
        "merchant_abn", "name", "segment_id", "segment", "dims_imputed", *dim_names
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write.mode("overwrite").parquet(str(OUTPUT_DIR / "merchant_behaviour_segments"))
    print(f"Written to {OUTPUT_DIR / 'merchant_behaviour_segments'}")

    print("Done.")


if __name__ == "__main__":
    main()
