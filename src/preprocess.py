import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler

def get_target_path(base_name):
    raw_dir = "data/raw"
    p1 = os.path.join(raw_dir, f"{base_name} (1).csv")
    p2 = os.path.join(raw_dir, f"{base_name}.csv")
    return p1 if os.path.exists(p1) else p2

def process_and_save(spark, input_path, output_name): # タイポ修正
    if not os.path.exists(input_path):
        print(f"Skipping: {input_path} not found.")
        return

    # 1. 読み込み
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    # 2. 特徴量カラム（id, target以外）を特定
    wavelenght_cols = [c for c in df.columns if c not in ["id", "target"]]
    # 3. VectorAssemblerによるベクトル化
    # 変数名をwavelenght_colsに統一
    assembler = VectorAssembler(inputCols=wavelenght_cols, outputCol="features")
    df_vector = assembler.transform(df)

    # 4. 必要な列だけ選択してParquetで保存
    output_path = f"data/processed/{output_name}.parquet"
    select_cols = ["id", "features"]
    if "target" in df.columns:
        select_cols.append("target")

    # 変数名をdf_vectorに修正
    df_vector.select(*select_cols).write.mode("overwrite").parquet(output_path)
    print(f"✅ Saved: {output_path}")

def main():
    # 24コアのMasterに接続！
    spark = SparkSession.builder \
        .master("spark://spark-master:7077") \
        .appName("Spectral-Preprocess-Pipeline") \
        .getOrCreate()
    
    os.makedirs("data/processed", exist_ok=True) # exist_okに修正

    process_and_save(spark, get_target_path("train"), "train_features")
    process_and_save(spark, get_target_path("test"), "test_features")

    spark.stop()

if __name__ == "__main__": # アンダーバー修正
    main()