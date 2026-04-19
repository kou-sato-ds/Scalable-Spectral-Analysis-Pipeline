import pandas as pd
import numpy as np
import os

def generate_dummy_data(output_path="data/raw/dummy_spectral_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    n_rows = 100
    n_features = 1556  # Pydanticで定義予定の次元数
    
    # 1. 正常なスペクトルデータの生成 (0.0 ~ 1.0 の範囲)
    data = np.random.uniform(0.1, 0.9, size=(n_rows, n_features))
    
    # 2. 樹種ラベルの生成 (不均衡を再現: Label_Aが極端に少ない)
    # Label_A: 2件, Label_B: 48件, Label_C: 50件
    labels = (["Label_A"] * 2) + (["Label_B"] * 48) + (["Label_C"] * 50)
    
    df = pd.DataFrame(data, columns=[f"feat_{i:04d}" for i in range(n_features)])
    df["tree_type"] = labels
    df["sample_id"] = range(n_rows)
    
    # --- 異常値の仕込み (テスト用) ---
    # 異常値1: 反射率が1.0を超えている (行 0, 列 0)
    df.iloc[0, 0] = 5.5
    
    # 異常値2: 欠損値 NaN が含まれている (行 1, 列 10)
    df.iloc[1, 10] = np.nan
    
    df.to_csv(output_path, index=False)
    print(f"✅ Dummy data generated at: {output_path}")
    print(f"📊 Shape: {df.shape} | Labels: {df['tree_type'].value_counts().to_dict()}")

if __name__ == "__main__":
    generate_dummy_data()