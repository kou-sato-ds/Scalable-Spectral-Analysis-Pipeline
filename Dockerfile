# 1. ベースイメージ：Python公式（軽量なDebian系）
FROM python:3.11-slim

# 2. 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    curl \
    procps \
    && apt-get clean

# 3. Sparkのインストール（高速ミラーサイトを優先）
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
RUN curl -fSL https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -o spark.tgz || \
    curl -fSL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -o spark.tgz \
    && tar -xzf spark.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark.tgz
    
# 4. 環境変数の設定
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH

# 🔥 Java 21の制限解除設定（これがないとエラーになります）
ENV JDK_JAVA_OPTIONS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED"

# ⚡️ Spark内部でArrowを最適化して使用する設定を事前注入（追加！）
ENV SPARK_CONF_DIR=$SPARK_HOME/conf
RUN echo "spark.sql.execution.arrow.pyspark.enabled true" >> $SPARK_HOME/conf/spark-defaults.conf

# 5. requirements.txtのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 作業ディレクトリの設定
WORKDIR /app
COPY . /app

# 7. コンテナ起動時のデフォルトコマンド
CMD ["python", "src/main.py"]