### 🚀 Spark 4.1.1 写経：Connect & DataFrame API 核心実装
# Spark Connectによるリモート接続と、公式が推奨する「DataFrame APIによる宣言的操作」を統合した型

"""
Apache Spark 4.1.1: Learning from Official Documentation
- Focus: Spark Connect (Client-Server Decoupling)
- Focus: Modern DataFrame API
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. Spark Connect 接続の構築 (SSの 'Spark Connect' セクションに対応)
# 従来の local ではなく、'sc://' (Spark Connectプロトコル) を使用するのがモダンな作法
spark = SparkSession.builder \
    .remote("sc://localhost:15002") \
    .appName("BigTech_Standard_Pipeline") \
    .getOrCreate()

try:
    # 2. データの読み込み (Parquet形式を推奨)
    # 1,500次元のデータを扱う際、必要なカラムだけを select するのが鉄則
    df = spark.read.parquet("data/processed/spectral_data.parquet")

    # 3. DataFrame API による関数チェーン (SSの 'Unified engine' に対応)
    # 読みやすく、並列実行計画（Logical Plan）が最適化されやすい書き方
    analysis_result = df.select("sample_id", "reflectance_val_100", "label") \
        .filter(F.col("reflectance_val_100") > 0.5) \
        .withColumn(
            "reflectance_status", 
            F.when(F.col("reflectance_val_100") > 0.8, "High").otherwise("Normal")
        ) \
        .groupBy("label", "reflectance_status") \
        .agg(F.count("sample_id").alias("total_count")) \
        .orderBy(F.desc("total_count"))

    # 4. 結果の表示
    analysis_result.show(truncate=False)

except Exception as e:
    # 実務レベルの例外処理
    print(f"Pipeline Error: {e}")

finally:
    # セッションのクローズ（リソース管理の徹底）
    spark.stop()

### 💡 写経の際のチェックポイント

#1.  **`sc://localhost:15002`**:「Decouples Spark client applications（クライアントとサーバーの分離）」を実現 最新のSpark
#2.  **`functions as F`**: 生のカラム名を文字列で書くのではなく、`F.col()` を使うことで、エンジンの最適化を助け、タイポによるバグを激減
#3.  **`select` の早期実行**:1,500次元あるうちの、必要な数次元だけを最初に選ぶ。クラウドでの課金を抑え、処理を高速化する「データエンジニアの規律」

## 正しいプロトコル記述

#spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()

# 正しい関数チェーン（カンマを忘れずに！）
#df.withColumn("new_col", F.when(F.col("val") > 0.8, "High").otherwise("Low"))