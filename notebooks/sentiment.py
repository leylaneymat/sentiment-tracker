from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType, StringType
from nltk.sentiment import SentimentIntensityAnalyzer

def main():
    spark = SparkSession.builder \
        .appName("MentalHealth_Sentiment") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()

    print("Spark session started")

    input_path = "hdfs:///data/processed/mental_health_clean.parquet"
    output_path = "hdfs:///data/analyzed/mental_health_with_sentiment.parquet"

    print(f"Reading processed data from: {input_path}")
    df = spark.read.parquet(input_path)

    print(f"Loaded {df.count():,} records")

    sia = SentimentIntensityAnalyzer()

    def get_sentiment_score(text):
        if text is None:
            return 0.0
        return float(sia.polarity_scores(text)["compound"])

    def get_sentiment_label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    sentiment_score_udf = udf(get_sentiment_score, FloatType())
    sentiment_label_udf = udf(get_sentiment_label, StringType())

    df = df.withColumn(
        "sentiment_score",
        sentiment_score_udf(col("post_text_clean"))
    ).withColumn(
        "sentiment_label",
        sentiment_label_udf(col("sentiment_score"))
    )

    print("Sentiment analysis done. Sample:")
    df.select(
        "post_id",
        "post_text_clean",
        "sentiment_score",
        "sentiment_label"
    ).show(5, truncate=60)

    df.coalesce(1) \
      .write \
      .mode("overwrite") \
      .option("compression", "snappy") \
      .parquet(output_path)

    print(f"Saved sentiment data to: {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()