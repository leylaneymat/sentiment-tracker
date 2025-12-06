from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, DoubleType
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, when, lower, trim, regexp_replace, sha2

def main():
    spark = SparkSession.builder \
        .appName("MentalHealth_ProcessOnly") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

    schema = StructType([
        StructField("row_index", IntegerType(), True),
        StructField("post_id", StringType(), True),
        StructField("post_created", StringType(), True),
        StructField("post_text", StringType(), True),
        StructField("user_id", LongType(), True),
        StructField("followers", IntegerType(), True),
        StructField("friends", IntegerType(), True),
        StructField("favourites", IntegerType(), True),
        StructField("statuses", IntegerType(), True),
        StructField("retweets", IntegerType(), True),
        StructField("label", IntegerType(), True)
    ])

    # Load from HDFS
    input_path = "hdfs:///data/raw_data/mental-health-data.csv" 
    df = spark.read \
        .option("header", "true") \
        .option("multiLine", "true") \
        .option("quote", '"') \
        .option("escape", '"') \
        .schema(schema) \
        .csv(input_path)
        
    print(f"Loaded {df.count():,} records")

    # Processing
    df = df \
        .withColumn("post_ts", to_timestamp(col("post_created"), "EEE MMM dd HH:mm:ss Z yyyy")) \
        .drop("post_created") \
        .withColumn("hour", hour("post_ts")) \
        .withColumn("day_of_week", dayofweek("post_ts")) \
        .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0)) \
        .withColumn("post_text_clean", lower(trim(regexp_replace(col("post_text"), "[^\\u0020-\\u007E\\u0080-\\uFFFF]", " ")))) \
        .withColumn("is_retweet", col("post_text").startswith("RT @").cast("byte")) \
        .withColumn("has_mental_health_keyword",
            col("post_text").rlike("(?i)#(anxiety|depression|mentalhealth|suicide|ptsd|bipolar)").cast("byte")
        ) \
        .withColumn("engagement_ratio", 
            when(col("followers") > 0, col("retweets") / col("followers")).otherwise(0.0)
        ) \
        .withColumn("user_hash", sha2(col("user_id").cast("string"), 256)) \
        .drop("user_id", "post_text")

    print("Processing done. Sample:")
    df.select("post_id", "post_ts", "post_text_clean", "label", "hour", "is_retweet").show(3, truncate=50)

    output_path = "hdfs:///data/processed/mental_health_clean.parquet"

    df.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(output_path)

    print(f"Saved as SINGLE file to: {output_path}")

    # Show output file info
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.defaultFS", "hdfs://namenode:9000")
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark.sparkContext._jsc.hadoopConfiguration())
    path = spark._jvm.org.apache.hadoop.fs.Path(output_path)
    if fs.exists(path):
        files = fs.listStatus(path)
        for f in files:
            if f.getPath().getName().startswith("part-"):
                print(f"📄 Output file: {f.getPath().getName()} | Size: {f.getLen()} bytes")

    spark.stop()

if __name__ == "__main__":
    main()