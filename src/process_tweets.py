import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

def create_spark_session():
    spark = SparkSession.builder \
        .appName("Sentiment Pre-processing") \
        .master("spark://spark-master:7077") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark

def clean_tweet_text(text_col):
    cleaned_col = F.lower(text_col)
    cleaned_col = F.regexp_replace(cleaned_col, r"http\S+", "")  # Remove URLs
    cleaned_col = F.regexp_replace(cleaned_col, r"@\S+", "")     # Remove @mentions
    cleaned_col = F.regexp_replace(cleaned_col, r"RT ", "")      # Remove "RT "
    cleaned_col = F.regexp_replace(cleaned_col, r"[^\w\s#]", "") # Remove punctuation (except for hashtags)
    cleaned_col = F.trim(cleaned_col)
    return cleaned_col

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
        print(f"Error reading from HDFS: {e}")
        spark.stop()
        return
    
    print("Raw data loaded.")

    print("Cleaning and transforming data...")

    try:
        working_df = raw_df.select(
            F.col("tweet_id"),
            F.col("created_at"),
            F.col("text").alias("original_text")
        )
    except pyspark.sql.utils.AnalysisException as e:
        print("ERROR: A column name is wrong in your schema.")
        print(f"Full error: {e}")
        spark.stop()
        return

    cleaned_df = working_df.withColumn(
        "cleaned_text", clean_tweet_text(F.col("original_text"))
    )
    
    transformed_df = cleaned_df.withColumn(
        "date",
        F.to_date(F.col("created_at"), "EEE MMM dd HH:mm:ss Z yyyy")
    )
    
    final_df = transformed_df.select(
        "tweet_id",
        "date",
        "original_text",
        "cleaned_text"
    ).filter(F.col("cleaned_text").isNotNull()) 

    print("--- SUCCESS: Data fully transformed ---")
    final_df.printSchema()
    final_df.show(5)

    spark.stop()

if __name__ == "__main__":
    main()