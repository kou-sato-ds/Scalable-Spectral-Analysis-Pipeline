# 1. ベースイメージ：Python公式（軽量なDebian系）
FROM python:3.11-slim

# 2. 必要なシステムパッケージのインストール（Java 21に変更）
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    curl \
    procps \
    && apt-get clean

# 3. Sparkのインストール（AWS EMR等で使われるバージョンを意識）
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
RUN curl -O https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# 4. 環境変数の設定
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH

# 5. Mohejiさんの最強requirements.txtをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 作業ディレクトリの設定
WORKDIR /app
COPY . /app

# 7. コンテナ起動時のデフォルトコマンド
CMD ["python", "src/main.py"]