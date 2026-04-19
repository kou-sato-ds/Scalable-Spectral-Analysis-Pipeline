# 📖 Operational Guide

本ドキュメントは、`Scalable-Spectral-Analysis-Pipeline` を安全かつ効率的に運用するための標準操作手順書（SOP）です。

---

## 1. 開発環境のクイックスタート

本プロジェクトは Docker によって抽象化されており、以下の手順でローカルに分散処理環境を構築できます。

### 🚀 手順
1. **イメージのビルド** Java 21 と Spark 3.5.0 が最適化された環境を構築します。
   ```bash
   docker build -t spectral-pipeline .
   ```

2. **コンテナの起動** ```bash
   docker run --rm -it spectral-pipeline
   ```

---

## 2. データ品質の検品（バリデーション）

新しいデータ（`data/raw/`）を追加した際は、必ず学習前に以下のバリデーションを実行し、データの整合性を確認してください。

### 🔍 実行コマンド
```bash
python tests/validate_data.py
```

### ✅ チェック項目
- **物理的範囲**: 反射率が 0.0 〜 1.0 の範囲に収まっているか。
- **欠損値**: 1,500次元超の全カラムに `NaN` が含まれていないか。
- **ラベル分布**: 特定の樹種が極端に少ない（5%未満）不均衡状態になっていないか。

---

## 3. トラブルシューティング

### ⚠️ JVM メモリエラー (`UnsupportedOperationException`)
Java 21 のカプセル化強化により、Spark がメモリにアクセスできずエラーが出る場合があります。
- **対策**: `Dockerfile` または実行時の `JDK_JAVA_OPTIONS` に以下のフラグが含まれているか確認してください。
  ```text
  --add-opens=java.base/java.nio=ALL-UNNAMED
  ```

### ⚠️ NumPy 2.x 互換性エラー
Python 3.14 環境では NumPy 2.0 以上が推奨されます。
- **対策**: `np.trapz` でエラーが出る場合は、`np.trapezoid` に置換されているか確認してください。

---

## 4. 運用ポリシー
- **ADRの更新**: アーキテクチャに大きな変更を加える際は、必ず `adr/` ディレクトリに新しい記録を追加してください。
- **コスト管理**: AWS Glue を使用する際は、`%idle_timeout 10` が設定されていることを確認し、リソースの垂れ流しを防止してください。