import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

def create_spark_session():
    """
    AWS Glueやローカル環境での実行を想定したSpark Sessionの初期化。
    """
    return SparkSession.builder \
        .appName("Scalable-Spectral-Analysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate() # get_all_or_createを修正

def main():
    # 1. Spark Sessionの起動
    spark = create_spark_session()
    print("Spark Session initialized successfully with AQE enabled.")

    # 2. スキーマ（型）の定義
    feature_fields = [
        StructField(f"feature_{i:03}", DoubleType(), True) for i in range(120)
    ]
    
    # メタデータと120個の特徴量を合体
    schema = StructType([
        StructField("sample_number", StringType(), True),
        StructField("species_number", StringType(), True),
    ] + feature_fields)

    print(f"Schema defined: Total fields = {len(schema.fields)}")

    # 3. データの読み込み（索敵開始）
    try:
        # data/train.csv を読み込みます
        train_df = spark.read.csv("data/train.csv", header=True, schema=schema)
        
        print(f"Total records loaded: {train_df.count()}")
        train_df.show(5) # 最初の5行を偵察

        # 4. 代表的な特徴量の統計量をスキャン
        print("Scanning data distribution...")
        train_df.select("feature_000", "feature_060", "feature_119").summary().show()

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Hint: Check if 'data/train.csv' exists.")
    
    print("Ready for large-scale spectral data processing.")

# 最後に main() を呼び出すだけ
if __name__ == "__main__":
    main()