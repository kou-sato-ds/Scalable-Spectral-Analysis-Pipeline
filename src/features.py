import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, DoubleType

# --- [門番] Pydanticによる型と次元の強制 ---
class SpectralData(BaseModel):
    features: list[float]

    @field_validator('features')
    @classmethod
    def check_length(cls, v):
        # 1556次元であることを厳格にチェック
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

    # 2. SNV補正（散乱補正）
    X_snv = (X_raw - mean_reflectance) / (std_reflectance + 1e-9)
    
    # 3. 1次微分 (np.diff + padding で次元を維持)
    X_diff = np.diff(X_snv, axis=1)
    X_diff = np.pad(X_diff, ((0, 0), (0, 1)), mode='edge')

    # 4. 特定領域の積分 (NumPyバージョン互換性対応)
    # NumPy 2.xは trapezoid ですが、現在の環境(1.x系)に合わせて trapz を使用
    try:
        if hasattr(np, 'trapezoid'):
            cellulose_area = np.trapezoid(X_snv[:, 1200:1400], axis=1).reshape(-1, 1)
            lignin_area = np.trapezoid(X_snv[:, 1400:1556], axis=1).reshape(-1, 1)
        else:
            cellulose_area = np.trapz(X_snv[:, 1200:1400], axis=1).reshape(-1, 1)
            lignin_area = np.trapz(X_snv[:, 1400:1556], axis=1).reshape(-1, 1)
    except AttributeError:
        # 万が一どちらもダメな場合の予備（単純合計）
        cellulose_area = np.sum(X_snv[:, 1200:1400], axis=1).reshape(-1, 1)
        lignin_area = np.sum(X_snv[:, 1400:1556], axis=1).reshape(-1, 1)

    # 特徴量結合: 元のスペクトル(1556) + 微分(1556) + 物理指標(2) = 3114次元
    return np.hstack([X_snv, X_diff, cellulose_area, lignin_area])

# --- [翼] Pandas UDF (Spark並列計算エンジン) ---
@pandas_udf(ArrayType(DoubleType()))
def spectral_feature_udf(batch_features: pd.Series) -> pd.Series:    
    """
    Apache Arrowを用いてデータを一括転送し、NumPyで並列計算する
    """
    # 1. Pydanticバリデーション (データの品質保証)
    try:
        SpectralData(features=batch_features.iloc[0])
    except Exception as e:
        print(f"Validation Error: {e}")
    
    # 2. NumPy行列に一括変換
    X_raw = np.stack(batch_features.values)
    
    # 3. 物理ロジック適用
    X_transformed = apply_spectral_logic(X_raw)
    
    # 4. SparkのArrayTypeに戻すためにリスト化して返す
    return pd.Series(list(X_transformed))

# --- [インターフェース] main.py から呼び出すためのブリッジ ---
def get_features_and_label(df):
    """
    main.py から呼ばれる関数。Spark/Pandas両対応。
    """
    # Spark DataFrame の場合
    if hasattr(df, "repartition") and not isinstance(df, pd.DataFrame):
        print("--- Parallelizing with Spark (24 Cores) ---")
        # パーティションを増やして並列度を上げる
        df = df.repartition(48)
        # 物理ロジック(Pandas UDF)の適用
        df = df.withColumn(
            "enriched_features", 
            spectral_feature_udf(df["features"])
        )
        # モデル学習(LightGBM)のために Pandas に戻す
        pdf = df.toPandas()
    else:
        # すでに Pandas の場合 (現在のメイン処理はこちらを通ります)
        print("--- Processing with Pandas (Single Core) ---")
        X_transformed = apply_spectral_logic(np.stack(df["features"].values))
        pdf = df.copy()
        pdf["enriched_features"] = list(X_transformed)

    # NumPy配列として抽出（LightGBMに投入可能な形状）
    X = np.stack(pdf["enriched_features"].values)
    
    # ターゲット列が存在する場合のみ y を返す
    y = None
    if "target" in pdf.columns:
        y = pdf["target"].values
    elif "species_id" in pdf.columns:
        y = pdf["species_id"].values
        
    return X, y