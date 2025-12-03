## Sentiment Tracker Project

Process and analyze mental-health–related tweets using Spark + HDFS + NLTK.

## 1\. Setup environment for sentiment analysis

This step prepares the environment to run the Spark jobs, ensuring necessary Python libraries and NLTK resources are available.

### Required dependencies (`requirements.txt`):

```
nltk
pyspark
```

### Setup commands:

  * Create and activate venv:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
  * Install dependencies:
    ```bash
    pip3 install -r requirements.txt
    ```
  * Download required NLTK resources:
    ```bash
    python3 -m nltk.downloader vader_lexicon punkt stopwords
    ```

## 2\. Create folders in HDFS

The command below creates the necessary folders, including `/data/processed` and `/data/sentiment` for job inputs/outputs, and sets permissions.

```bash
docker exec -it namenode bash -c "                        
  hdfs dfs -mkdir -p /data/processed /data/sentiment /data/raw_data && \
  hdfs dfs -chmod -R 777 /data
"
```

## 3\. Upload dataset to HDFS

This uploads the raw data file to the HDFS `/data/raw_data/` directory.

```bash
docker exec -it namenode hdfs dfs -put /app/raw_data/mental-health-data.csv /data/raw_data/
docker exec -it namenode hdfs dfs -ls /data/raw_data/
```

## 4\. Run preprocessing Spark job

Submit the Spark job to clean and process the raw data.

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
  /app/notebooks/processing.py
```

Output location:

```
hdfs:///data/processed/mental_health_clean.parquet
```

## 5\. Run sentiment analysis

Submit the Spark job that reads the cleaned data, performs sentiment analysis using NLTK, and writes the results.

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
  /app/notebooks/sentiment.py
```

Input location:

```
hdfs:///data/processed/mental_health_clean.parquet
```

Output location:

```
hdfs:///data/sentiment/
```
