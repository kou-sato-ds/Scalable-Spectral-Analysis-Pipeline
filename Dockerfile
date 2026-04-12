# 1. ベースイメージ：Python 3.11（安定版）
FROM python:3.11-slim

# 2. 必要なシステムパッケージのインストール
# Java 21、並列計算用の libgomp1、構築用の curl などを統合
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    curl \
    procps \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Spark 3.5.0 のインストール
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark

RUN curl -fSL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -o spark.tgz \
    && tar -xzf spark.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} \
    && rm spark.tgz

# 4. 環境変数の設定
ENV PATH=$PATH:$SPARK_HOME/bin

# Py4Jのパス解決（ワイルドカードによるエラーを避けるためシンボリックリンクを作成）
RUN ln -s $(ls $SPARK_HOME/python/lib/py4j-*-src.zip) $SPARK_HOME/python/lib/py4j-src.zip
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-src.zip:$PYTHONPATH

# 🔥 Java 21 のカプセル化（メモリ制限）を突破する設定
# これにより Spark/Arrow が java.nio にアクセス可能になります
ENV JDK_JAVA_OPTIONS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED"

# ⚡ Spark内部で Arrow をデフォルトで有効化する設定を注入
ENV SPARK_CONF_DIR=$SPARK_HOME/conf
RUN echo "spark.sql.execution.arrow.pyspark.enabled true" >> $SPARK_HOME/conf/spark-defaults.conf

# 5. Python 依存ライブラリのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 作業ディレクトリの設定とコピー
WORKDIR /app
COPY . /app

# 7. 実行コマンド
# コンテナ起動時にメイン処理を実行
CMD ["python", "src/main.py"]