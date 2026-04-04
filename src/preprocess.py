import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

def get_target_path(base_name):
    """(1)が付いているファイルと付いていないファイルの両方を探す"""
    raw_dir = "data/raw"
    p1 = os.path.join(raw_dir, f"{base_name} (1).csv")
    p2 = os.path.join(raw_dir, f"{base_name}.csv")
    return p1 if os.path.exists(p1) else p2

def main():
    spark = SparkSession.builder.appName("Spectral-Preprocess").getOrCreate()
    
    # 読み込み
    train_path = get_target_path("train")
    test_path = get_target_path("test")
    
    # 保存先ディレクトリの作成
    os.makedirs("data/processed", exist_ok=True)
    
    # --- ここに前述のprocess_and_saveロジックが入る ---
    # 保存先を data/processed/ に指定
    # process_and_save(spark, train_path, "data/processed/processed_train")
    # process_and_save(spark, test_path, "data/processed/processed_test")
    
    spark.stop()