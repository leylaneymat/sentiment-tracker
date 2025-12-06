import sys

from pyspark.sql import SparkSession


def main():
    print("Starting export script...")

    try:
        # Create Spark session
        spark = (
            SparkSession.builder.appName("ExportToCSV")
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
            .getOrCreate()
        )

        print("Spark session created")

        input_path = (
            "hdfs://namenode:9000/data/analyzed/mental_health_with_sentiment.parquet"
        )
        output_path = "hdfs://namenode:9000/data/export/sentiment_results.csv"

        print(f"Reading from: {input_path}")

        # Read parquet
        df = spark.read.parquet(input_path)

        count = df.count()
        print(f"Loaded {count:,} records")

        if count == 0:
            print("WARNING: DataFrame is empty!")
            spark.stop()
            sys.exit(1)

        # Show sample
        print("\nSample data:")
        df.show(5, truncate=50)

        # Export to CSV with proper escaping and quoting
        print(f"\nExporting to: {output_path}")
        df.coalesce(1).write.mode("overwrite").option("header", "true").option(
            "quote", '"'
        ).option("escape", '"').option("quoteAll", "true").csv(output_path)

        print("✓ Export complete!")

        spark.stop()

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
