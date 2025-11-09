# Bitnami Spark image
FROM apache/spark-py:v3.4.0

# Switch to the root user
USER root

# Install all the Python library dependecies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Create a Hadoop config directory
RUN mkdir -p /etc/hadoop/conf
ENV HADOOP_CONF_DIR=/etc/hadoop/conf

# Switch back to user
USER 1001
