import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List

# --- 1. Pydantic による行単位の厳格な定義 ---
class SpectralRecord(BaseModel):
    sample_id: int
    tree_type: str
    # 1,500次元超の数値をリストとして受け取り、一括バリデーション
    features: List[float]

    @field_validator("features")
    @classmethod
    def check_values_range(cls, v: List[float]) -> List[float]:
        for x in v:
            # 欠損値(NaN)のチェック
            if np.isnan(x):
                raise ValueError("NaN detected in spectral data")
            # 物理的範囲(0-1)のチェック
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"Value {x} is out of physical range (0.0-1.0)")
        return v

# --- 2. バリデーション実行クラス ---
def run_validation(file_path="data/raw/dummy_spectral_data.csv"):
    df = pd.read_csv(file_path)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    errors = []

    print(f"🔍 Starting validation for: {file_path}")

    # A. Pydantic による「守り」のチェック (1 & 2)
    for i, row in df.iterrows():
        try:
            SpectralRecord(
                sample_id=row["sample_id"],
                tree_type=row["tree_type"],
                features=row[feature_cols].tolist()
            )
        except ValidationError as e:
            errors.append(f"Row {i} Validation Error: {e.json()}")

    # B. 戦略的な分布チェック (3)
    label_counts = df["tree_type"].value_counts(normalize=True)
    for label, ratio in label_counts.items():
        if ratio < 0.05:  # 5%未満を警告
            errors.append(f"Distribution Warning: Label '{label}' is too rare ({ratio:.2%})")

    # --- 3. 結果の出力 ---
    if errors:
        print("\n❌ Validation Failed with following issues:")
        for err in errors:
            print(f"  - {err}")
        # CIを落とすために例外を投げる、あるいは終了コード1を返す
        # raise SystemExit(1) 
    else:
        print("\n✅ All data quality checks passed!")

if __name__ == "__main__":
    run_validation()