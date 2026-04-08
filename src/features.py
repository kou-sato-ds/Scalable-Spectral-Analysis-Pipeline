import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql import SparkSession

# --- [門番] Pydanticによる型と次元の強制 ---
class SpectralData(BaseModel):
    features: list[float]

    @field_validator('features')
    @classmethod
    def check_length(cls, v):
        if len(v) != 1556:
            raise ValueError(f'スペクトル次元が不正です({len(v)}次元)。')
        return v

# --- [核] 物理ロジック本体 ---
def apply_spectral_logic(X_raw: np.ndarray) -> np.ndarray:    
    """
    SNV補正、1次微分、特定吸収帯の抽出を行う
    """
    # 1. 統計指標
    mean_reflectance = np.mean(X_raw, axis=1, keepdims=True)
    std_reflectance = np.std(X_raw, axis=1, keepdims=True)

    # 2. SNV補正
    X_snv = (X_raw - mean_reflectance) / (std_reflectance + 1e-9)
    # 3. 1次微分 (np.diff + padding)
    X_diff = np.diff(X_snv, axis=1)
    X_diff = np.pad(X_diff, ((0, 0), (0, 1)), mode='edge')

    # 4. 特定領域の積分 (NumPy 2.x対応: trapezoid)
    cellulose_area = np.trapezoid(X_snv[:, 1200:1400], axis=1).reshape(-1, 1)
    lignin_area = np.trapezoid(X_snv[:, 1400:1556], axis=1).reshape(-1, 1)

    # 特徴量結合
    return np.hstack([X_snv, X_diff, cellulose_area, lignin_area])

# --- [翼] Pandas UDF (これが24コアを回すエンジン) ---
@pandas_udf(ArrayType(DoubleType()))
def spectral_feature_udf(batch_features: pd.Series) -> pd.Series:    
    """
    Apache Arrowを用いてデータを一括転送し、NumPyで並列計算する
    """
    # 1. Pydanticバリデーション (先頭1件で代表チェック)
    SpectralData(features=batch_features.iloc[0])
    
    # 2. NumPy行列に変換
    X_raw = np.stack(batch_features.values)
    
    # 3. 物理ロジック適用
    X_transformed = apply_spectral_logic(X_raw)
    
    # 4. Sparkに戻すためにリスト化
    return pd.Series(list(X_transformed))

# --- [インターフェース] main.py から呼び出すためのブリッジ ---
def get_features_and_label(df):
    """
    main.py から呼ばれる関数。
    """
    # もし渡された df が Spark DataFrame なら repartition を実行
    # (AttributeError を防ぐための安全策)
    if hasattr(df, "repartition"):
        print("--- Parallelizing with Spark (24 Cores) ---")
        df = df.repartition(48)
        # 物理ロジック(Pandas UDF)の適用
        df = df.withColumn(
            "enriched_features", 
            spectral_feature_udf(df["features"])
        )
        # 最後に Pandas に変換してモデル学習へ渡す
        pdf = df.toPandas()
    else:
        # すでに Pandas の場合は Spark を通さずに処理（小規模データ用）
        print("--- Processing with Pandas (Single Core) ---")
        # 直接 UDF の中身のロジックを呼ぶ
        X_transformed = apply_spectral_logic(np.stack(df["features"].values))
        pdf = df.copy()
        pdf["enriched_features"] = list(X_transformed)

    # NumPy配列として返す（LightGBMが食べやすい形に）
    X = np.stack(pdf["enriched_features"].values)
    y = pdf["target"].values
    return X, y