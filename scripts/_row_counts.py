import glob
import os

TABLES_DIR = os.path.expanduser("~/MAST30034_Python/project2/data/tables")

for path in sorted(glob.glob(os.path.join(TABLES_DIR, "*"))):
    name = os.path.basename(path)
    if name.endswith(".csv"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f) - 1
        print(f"{name}: {n:,} rows (csv)")
    elif os.path.isdir(path):
        part_files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        print(f"{name}/: {len(part_files)} part-files (need pyarrow/spark for row count)")
    else:
        print(f"{name}: single parquet file (need pyarrow/spark for row count)")
