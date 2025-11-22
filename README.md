# Sentiment Tracker Project

Process mental-health–related tweets using Spark + HDFS.

## 1. Create folders in HDFS

```bash
docker exec -it namenode bash -c "                        
  hdfs dfs -mkdir -p /data/processed && \
  hdfs dfs -chmod -R 777 /data
"
```

## 2. Upload dataset to HDFS

```bash
docker exec -it namenode hdfs dfs -put /app/raw_data/mental-health-data.csv /data/raw_data/
docker exec -it namenode hdfs dfs -ls /data/raw_data/
```

## 3. Run preprocessing Spark job

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
  /app/notebooks/processing.py
```

Output location:
```
/data/processed/
```