import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from features import get_features_and_label
from pyspark.sql import SparkSession

def main():
    print("--- 🔥 EXECUTING FINISH BUSTER: SPARK × DOMAIN-FE ---")
    
    # 1. SparkSessionの構築（Java 21/Arrow制約をコード側でもケア）
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("UltimateEnsemble") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()

    # パス設定
    PROC_DIR = "data/processed"
    RAW_DIR = "data/raw"

    # 2. データの読み込み
    # Sparkで読み込み、モデル学習のためにPandasに変換
    # ※Arrowをオフにしているので、Javaのメモリ制限エラーを回避できます
    sdf = spark.read.parquet(os.path.join(PROC_DIR, "train_with_folds.parquet"))
    df = sdf.toPandas() 
    
    all_models = []
    test_probs = []

    # 3. 5-Fold 並列的思考学習
    for fold in range(5):
        print(f"\n--- FOLD {fold}: Analyzing Chemical Bonds... ---")
        
        # データの分割
        train_df = df[df['fold'] != fold].copy()
        val_df = df[df['fold'] == fold].copy()
        
        # 特徴量抽出
        X_train, y_train = get_features_and_label(train_df)
        X_val, y_val = get_features_and_label(val_df)
        
        # LightGBM パラメータ
        params = {
            'objective': 'multiclass',
            'num_class': 51,
            'metric': 'multi_logloss',
            'learning_rate': 0.02,
            'feature_fraction': 0.5,
            'num_leaves': 127,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'seed': 42 + fold,
            'verbosity': -1
        }
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        model = lgb.train(
            params, lgb_train,
            valid_sets=[lgb_train, lgb_val],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
        all_models.append(model)

    # 4. 究極推論 (アンサンブル)
    test_df = pd.read_parquet(os.path.join(PROC_DIR, "processed_test.parquet"))
    X_test, _ = get_features_and_label(test_df)
    
    # 学習時とテスト時の特徴量次元を統一
    target_dim = X_train.shape[1]
    if X_test.shape[1] < target_dim:
        X_test = np.pad(X_test, ((0,0), (0, target_dim - X_test.shape[1])), mode='constant')
    else:
        X_test = X_test[:, :target_dim]

    # 全モデルの知恵を平均（Soft Voting）
    for m in all_models:
        test_probs.append(m.predict(X_test))
    
    final_probs = np.mean(test_probs, axis=0)
    final_preds = np.argmax(final_probs, axis=1)

    # 5. 提出ファイルの生成
    sample_path = os.path.join(RAW_DIR, "sample_submit.csv")
    template = pd.read_csv(sample_path, header=None)
    
    pred_df = pd.DataFrame({"id": test_df["id"], "species_id": final_preds})
    
    # テンプレートのID順序を維持してマージ
    final_sub = pd.DataFrame({0: template[0]})
    final_sub = final_sub.merge(pred_df, left_on=0, right_on="id", how="left")
    final_sub["species_id"] = final_sub["species_id"].fillna(0).astype(int)

    # 指定形式(headerなし, indexなし)で保存
    final_sub[[0, "species_id"]].to_csv(
        os.path.join(RAW_DIR, "submission_ultimate.csv"), 
        index=False, 
        header=False
    )
    
    print("\n✅ FINISH BUSTER COMPLETE. CHECK data/raw/submission_ultimate.csv")
    spark.stop()

if __name__ == "__main__":
    main()