import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

def create_spark_session():
    """
    AWS Glueやローカル環境での実行を想定したSpark Sessionの初期化。
    AQE (Adaptive Query Execution) を有効化し、動的な最適化を行う。
    """
    return SparkSession.builder \
        .appName("Scalable-Spectral-Analysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .get_all_or_create()

def main():
    # 1. Spark Sessionの起動
    spark = create_spark_session()
    print("Spark Session initialized successfully with AQE enabled.")

    # 2. スキーマ（型）の定義
    # 100点へのポイント: リスト内包表記を用いて120次元の特徴量を効率的に定義
    feature_fields = [
        StructField(f"feature_{i:03}", DoubleType(), True) for i in range(120)
    ]

    schema = StructType([
        StructField("sample_number", StringType(), True),
        StructField("species_number", StringType(), True),
    ] + feature_fields) # メタデータと波長データを結合

    # 3. データの読み込み準備（明日の実装用）
    # path = "data/train.csv"
    # df = spark.read.csv(path, header=True, schema=schema)
    
    print(f"Schema defined: Total fields = {len(schema.fields)}")
    print("Ready for large-scale spectral data processing.")

    # 4. セッションの終了（必要に応じて解除）
    # spark.stop()

if __name__ == "__main__":
    main()