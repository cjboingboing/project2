from pathlib import Path
from pyspark.sql import SparkSession, functions as F
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "tables"

spark = (
    SparkSession.builder.appName("MAST30034 Project 2 -- merchant loading")
    .config("spark.sql.repl.eagerVal.enabled", True)
    .config("spark.sql.parquet.cacheMetadata", True)
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .getOrCreate()
)

def main():

    pass

if __name__ == "__main__":
    main()
