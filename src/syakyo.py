# pydantic features.py

from pydantic import BaseModel, field_validator # 最新のPydantic v2ではfield_validatorが推奨

class SpectralData(BaseModel):
    features: list[float]

    @field_validator('features')
    @classmethod
    def check_length(cls, v):
        # 1556次元でない不純なデータを通さない「鉄壁の守り」
        if len(v) != 1556:
            raise ValueError('スペクトル次元が不正です。測定データを確認してください。')
        return v

#PySpark SQL & UDF

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# 物理的な補正ロジック（例：温度補正や散乱補正の係数）
def apply_physical_correction(val):
    # 1行ずつの処理を書くだけで、Sparkが裏で全CPUに配ってくれます
    return val * 1.05 

# Sparkに「これは並列実行できる関数だよ」と登録する
correction_udf = udf(apply_physical_correction, DoubleType())

# 全データに一括適用
df = df.withColumn("corrected_val", correction_udf(df["raw_val"]))