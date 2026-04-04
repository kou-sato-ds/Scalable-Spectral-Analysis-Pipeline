import sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def create_spark_session():
    # Windowsでのエラーを抑制する設定を追加
    return SparkSession.builder \
        .appName("Spectral-KFold-Split") \
        .config("spark.driver.extraJavaOptions", "-Dcom.sun.management.jmxremote") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

def main():
    spark = create_spark_session()
    print("--- STARTING STRATIFIED K-FOLD SPLIT ---")

    # 1. 【修正】Sparkで直接読まず、Pandasで読んでからSparkに渡す
    input_path = "data/processed_train.parquet"
    print(f"Reading {input_path} via Pandas...")
    
    pdf = pd.read_parquet(input_path)
    df = spark.createDataFrame(pdf) # Pandas -> Spark
    
    # 2. 分割数（K）の設定
    K = 5
    
    # 3. 層化分割のロジック (ここは変更なし)
    window_spec = Window.partitionBy("species_id").orderBy(F.rand(seed=42))
    df_with_fold = df.withColumn("row_num", F.row_number().over(window_spec)) \
                     .withColumn("fold", (F.col("row_num") % K)) \
                     .drop("row_num")

    # 4. バランス確認
    print("--- FOLD DISTRIBUTION ---")
    df_with_fold.groupBy("fold").count().orderBy("fold").show()

    # 5. 保存
    output_path = "data/train_with_folds.parquet"
    print(f"--- SAVING SPLIT DATA TO {output_path} ---")
    
    # Spark -> Pandas に戻してから保存（Windowsエラー回避の鉄板）
    final_pd = df_with_fold.toPandas()
    final_pd.to_parquet(output_path, index=False)
    
    print(f"Success! Data with folds saved to: {output_path}")
    spark.stop()

if __name__ == "__main__":
    main()