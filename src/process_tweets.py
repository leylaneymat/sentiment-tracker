import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType

def create_spark_session():
    spark = SparkSession.builder \
        .appName("Sentiment Pre-processing") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    return spark

def main():
    spark = create_spark_session()
    print("SparkSession created. Connected to Spark Master.")

    HDFS_RAW_PATH = "hdfs://namenode:9000/project/raw/tweets.csv"

    print(f"Attempting to read raw CSV data from {HDFS_RAW_PATH}...")
    
    schema = StructType([
        StructField("tweet_id", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("text", StringType(), True),
    ])

    try:
        raw_df = spark.read.csv(HDFS_RAW_PATH, schema=schema, header=True)
    except Exception as e:
        print(f"Error reading from HDFS. Does the file exist at '{HDFS_RAW_PATH}'?")
        print(f"Error: {e}")
        spark.stop()
        return

    print("SUCCESS: Raw data loaded")
    raw_df.printSchema()
    raw_df.show(5, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()