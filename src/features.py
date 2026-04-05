import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField

# --- [門番] Pydanticによる型と次元の強制 ---
class SpectralData(BaseModel):
    features: list[float]

    @field_validator('features')
    @classmethod
    def check_length(cls, v):
        if len(v) != 1556:
            raise ValueError(f'スペクトル次元が不正です({len(v)}次元)。1556次元である必要があります。')
        return v

# --- [核] 物理ロジック本体 (ベクトル/行列両対応) ---
def apply_spectral_logic(X_raw):
    """
    物理的背景に基づく特徴量抽出
    X_raw: np.array (1次元または2次元)
    """
    # 2次元行列として扱うための調整
    is_1d = X_raw.ndim == 1
    if is_1d:
        X_raw = X_raw.reshape(1, -1)

    # 1. 統計指標 (散乱補正の基礎)
    mean_reflectance = np.mean(X_raw, axis=1, keepdims=True)
    std_reflectance = np.std(X_raw, axis=1, keepdims=True)

    # 2. SNV補正 (光のムラを除去)
    X_snv = (X_raw - mean_reflectance) / (std_reflectance + 1e-9)

    # 3. 1次微分 (SG法の簡易版として波形を際立たせる)
    X_diff = np.diff(X_snv, axis=1)
    X_diff = np.pad(X_diff, ((0, 0), (0, 1)), mode='edge')

    # 4. 特定吸収帯 (OH基: 水分/セルロース)
    oh_1450 = np.mean(X_snv[:, 300:400], axis=1, keepdims=True)
    oh_1940 = np.mean(X_snv[:, 800:900], axis=1, keepdims=True)
    oh_ratio = oh_1450 / (oh_1940 + 1e-9)

    # 5. 局所積分 (成分エリアの面積)
    # NumPy 2.0+ 推奨の np.trapezoid を使用
    cellulose_area = np.trapezoid(X_snv[:, 1200:1400], axis=1).reshape(-1, 1)
    lignin_area = np.trapezoid(X_snv[:, 1400:1556], axis=1).reshape(-1, 1)

    # 特徴量結合
    X_combined = np.hstack([
        X_snv, X_diff, mean_reflectance, std_reflectance, 
        oh_ratio, cellulose_area, lignin_area
    ])

    return X_combined.flatten() if is_1d else X_combined

# --- [翼] Spark用UDFの登録 ---
# 特徴量の総次元数を計算して戻り値の型を定義
# (1556*2 + 5 = 3117次元)
spectral_udf = udf(lambda x: apply_spectral_logic(np.array(x)).tolist(), ArrayType(DoubleType()))

def get_features_and_label(pdf: pd.DataFrame):
    # Pydanticによる一括バリデーション (実務での信頼性アピール)
    for feat in pdf['features']:
        SpectralData(features=feat)
        
    X_raw = np.stack(pdf['features'].values)
    X_combined = apply_spectral_logic(X_raw)
    
    y = pdf['species_id'].values if 'species_id' in pdf.columns else None
    return X_combined, y