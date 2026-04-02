from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

def create_spark_session():
    """
    AWS Glueやローカル環境での実行を想定したSpark Sessionの初期化。
    """
    return SparkSession.builder \
        .appName("Scalable-Spectral-Analysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .get_all_or_create()

def main():
    spark = create_spark_session()
    
    # 100点へのポイント: スキーマ（型）を明示的に定義する準備
    # スペクトルデータ(feature_000-119)はDoubleType（浮動小数点）で定義
    schema = StructType([
        StructField("sample_number", StringType(), True),
        StructField("species_number", StringType(), True),
        # ここに波長データの定義が続く...
    ])
    
    print("Spark Session initialized successfully.")
    # spark.stop() # ローカルテスト時はコメントアウト解除

if __name__ == "__main__":
    main()