import os
from pyspark.sql import SparkSession

BASE = os.path.expanduser("~/MAST30034_Python/project2/data/tables")

spark = (
    SparkSession.builder.appName("row_counts")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

def count_and_schema(name, path):
    df = spark.read.parquet(path)
    n = df.count()
    print(f"{name}: {n:,} rows, {len(df.columns)} cols -> {df.columns}")

count_and_schema("tbl_merchants.parquet", os.path.join(BASE, "tbl_merchants.parquet"))
count_and_schema("consumer_user_details.parquet", os.path.join(BASE, "consumer_user_details.parquet"))

snapshots = [
    os.path.join(BASE, "transactions_20210228_20210827_snapshot"),
    os.path.join(BASE, "transactions_20210828_20220227_snapshot"),
    os.path.join(BASE, "transactions_20220228_20220828_snapshot"),
]
for s in snapshots:
    count_and_schema(os.path.basename(s), s)

print("--- combined (union of all 3 snapshots, read as one) ---")
dfs = [spark.read.parquet(s) for s in snapshots]
df_all = dfs[0]
for d in dfs[1:]:
    df_all = df_all.unionByName(d)
total = df_all.count()
distinct_order_ids = df_all.select("order_id").distinct().count()
distinct_full_rows = df_all.distinct().count()
print(f"all_transactions: {total:,} rows, {len(df_all.columns)} cols -> {df_all.columns}")
print(f"distinct order_id: {distinct_order_ids:,} (dupes if < total)")
print(f"distinct full rows: {distinct_full_rows:,} (dupes if < total)")

spark.stop()
