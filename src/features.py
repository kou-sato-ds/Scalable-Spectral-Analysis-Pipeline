import numpy as np
import pandas as pd

def get_features_and_label(pdf: pd.DataFrame):
    # 1. 基本スペクトルの展開 (1556次元)
    X_raw = np.stack(pdf['features'].values)
    
    # --- [ドメイン知識 1] 散乱（Scattering）指標の抽出 ---
    # 全体平均と標準偏差は、木材の密度や表面粗さを反映する
    mean_reflectance = np.mean(X_raw, axis=1, keepdims=True)
    std_reflectance = np.std(X_raw, axis=1, keepdims=True)
    
    # --- [ドメイン知識 2] SNV補正 ---
    X_snv = (X_raw - mean_reflectance) / std_reflectance
    
    # --- [ドメイン知識 3] 1次微分 (波形の傾き) ---
    X_diff = np.diff(X_snv, axis=1)
    X_diff = np.pad(X_diff, ((0, 0), (0, 1)), mode='edge')
    
    # --- [ドメイン知識 4] 特定吸収帯の抽出 (OH基 / セルロース) ---
    # 波長インデックスを推定（1556次元が約1100-2500nmに対応と仮定）
    # 1450nm付近(idx: 300-400), 1940nm付近(idx: 800-900)などを抽出
    # ※正確な波長対応が不明な場合でも、統計的なピーク付近を捉える
    oh_1450 = np.mean(X_snv[:, 300:400], axis=1, keepdims=True)
    oh_1940 = np.mean(X_snv[:, 800:900], axis=1, keepdims=True)
    
    # 含水率レンジの非線形性を捉える「比率」特徴量
    oh_ratio = oh_1450 / (oh_1940 + 1e-9)
    
    # --- [ドメイン知識 5] 化学成分エリアの局所積分 ---
    # NumPy 2.0以降や最新版ではこちらが推奨
    cellulose_area = np.trapezoid(X_snv[:, 1200:1400], axis=1).reshape(-1, 1)
    lignin_area = np.trapezoid(X_snv[:, 1400:1556], axis=1).reshape(-1, 1)

    # 全てを結合 (生 + 微分 + 物理統計量)
    X_combined = np.hstack([
        X_snv, 
        X_diff, 
        mean_reflectance, # 散乱強度
        std_reflectance,  # 表面粗さ
        oh_ratio,         # 水分非線形
        cellulose_area,   # セルロース
        lignin_area       # リグニン
    ])
    
    y = pdf['species_id'].values if 'species_id' in pdf.columns else None
    return X_combined, y