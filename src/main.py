import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from features import get_features_and_label
from pyspark.sql import SparkSession

def main():
    print("--- 🔥 EXECUTING FINISH BUSTER: SPARK × DOMAIN-FE ---")
    
    # Sparkの本気: ローカルリソースをフル活用
    spark = SparkSession.builder.master("local[*]").appName("UltimateEnsemble").getOrCreate()

    # 1. データの読み込み
    PROC_DIR = "data/processed"
    RAW_DIR = "data/raw"

    # 修正前: df = spark.read.parquet("data/processed/train_features.parquet")
    df = spark.read.parquet("data/processed/train_with_folds.parquet")    
    all_models = []
    test_probs = []

    # 2. 5-Fold 並列的思考学習
    for fold in range(5):
        print(f"\n--- FOLD {fold}: Analyzing Chemical Bonds... ---")
        train_pdf = df[df['fold'] != fold]
        val_pdf = df[df['fold'] == fold]
        
        X_train, y_train = get_features_and_label(train_pdf)
        X_val, y_val = get_features_and_label(val_pdf)
        
        # 物理知識を最大限活かすパラメータ
        params = {
            'objective': 'multiclass',
            'num_class': 51,
            'metric': 'multi_logloss',
            'learning_rate': 0.02, # 慎重に微細な差を学習
            'feature_fraction': 0.5, # 物理特徴量が多いので多様性を確保
            'num_leaves': 127,      # 複雑な非線形関係を許容
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'lambda_l1': 0.1,       # 不要な波長を落とす
            'seed': 42 + fold
        }
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        model = lgb.train(
            params, lgb_train,
            valid_sets=[lgb_train, lgb_val],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
        )
        all_models.append(model)

    # 3. 究極推論 (アンサンブル)
    test_df = pd.read_parquet(os.path.join(PROC_DIR, "processed_test.parquet"))
    X_test, _ = get_features_and_label(test_df)
    
    # 次元調整 (ドメイン特徴量追加後も対応)
    target_dim = X_train.shape[1]
    X_test = np.pad(X_test, ((0,0), (0, max(0, target_dim - X_test.shape[1]))), mode='constant')[:, :target_dim]

    # 全モデルの知恵を平均
    for m in all_models:
        test_probs.append(m.predict(X_test))
    
    final_preds = np.argmax(np.mean(test_probs, axis=0), axis=1)

    # 4. 提出ファイルの生成
    sample_path = os.path.join(RAW_DIR, "sample_submit.csv")
    template = pd.read_csv(sample_path, header=None)
    pred_df = pd.DataFrame({"id": test_df["id"], "species_id": final_preds})
    final_sub = pd.DataFrame({"id": template[0]}).merge(pred_df, on="id", how="left")
    final_sub["species_id"] = final_sub["species_id"].fillna(0).astype(int)

    final_sub[["id", "species_id"]].to_csv(os.path.join(RAW_DIR, "submission_ultimate.csv"), index=False, header=False)
    print("\n✅ FINISH BUSTER COMPLETE. CHECK data/raw/submission_ultimate.csv")
    spark.stop()

if __name__ == "__main__":
    main()