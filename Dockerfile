# Bitnami Spark image
FROM apache/spark-py:v3.4.0

# Switch to the root user
USER root

# Install all the Python library dependecies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN python3 -m nltk.downloader -d /opt/spark/nltk_data vader_lexicon punkt stopwords

# Create a Hadoop config directory
RUN mkdir -p /etc/hadoop/conf
ENV HADOOP_CONF_DIR=/etc/hadoop/conf

# Create Spark work directory with proper permissions
RUN mkdir -p /opt/spark/work && \
    chown -R 1001:1001 /opt/spark/work && \
    chmod -R 755 /opt/spark/work

# Switch back to user
USER 1001
