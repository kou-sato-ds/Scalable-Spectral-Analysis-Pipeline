# # Scalable Spectral Analysis Pipeline 🌲🌾

![Python CI](https://github.com/kou-sato-ds/Scalable-Spectral-Analysis-Pipeline/actions/workflows/ci.yml/badge.svg)

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-3.5.0-orange?logo=apachespark)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📝 Overview

本プロジェクトは、SIGNATEの「近赤外研究会 スペクトル分析チャレンジ」を題材に、高次元なスペクトルデータから樹種を分類するパイプラインを構築したものです。
単なるモデル構築に留まらず、**木材科学のドメイン知識**を特徴量エンジニアリングに融合させ、**Apache Spark** による分散並列処理基盤（24コア活用）を統合した、実務直結型のエンジニアリング手法を採用しています。

## 🏗️ Architecture & Pipeline

1.  **Distributed Preprocessing**: PySparkを用いた1,500次元超のスペクトルデータのベクトル化・Parquet変換。
2.  **Domain Feature Engineering**: 近赤外分光法(NIR)の物理的特性に基づき、SNV補正や1次微分を実装。
3.  **Scalable Training**: LightGBMを用いた 5-fold CV。Sparkを活用したスケーラブルな分散学習。
4.  **Robust Inference Engine**: 訓練/テストデータの次元不一致補正、およびテンプレートマージによる完全な提出フォーマット保証。
5.  **Modern JVM Infrastructure**: **Java 21** 世代の厳格なメモリ保護を突破する JVM チューニングを施した Docker 基盤。

### 🛡️ Robust Data Validation

  - **Pydantic Guard**: 1,556次元のスペクトル形状を厳格にチェック。
  - **Schema Enforcement**: 分散処理の各ステージで、型の不整合や次元の欠落を許さない堅牢なパイプライン。
  - **Hybrid Processing**: Sparkの分散力と、NumPyの高度な物理演算を「Pandas UDF (Apache Arrow)」で融合。

### 🔧 Infrastructure & Scalability

  - **Java 21 & Spark 3.5.0**: 最新の長期サポート（LTS）世代を採用。内部メモリアクセス制限（`UnsupportedOperationException`）を、`--add-opens` オプションのインジェクションにより解決。
  - **Volume Mounting**: Docker Composeを利用し、ホスト側の資産とコンテナをリアルタイム同期。
  - **24-Core Optimization**: ローカルリソースをフルに使い切るスレッド設計により、大規模な特徴量生成を高速化。

### 📊 Cluster Status
![Spark UI](./Images/spark_ui.png)
*Local Spark Cluster: 1 Master, 2 Workers (Total 24 Cores) 稼働中*

![Spark UI 24 Cores](./Images/spark_ui_24cores.png)
*Local Spark Cluster: 1 Master, 2 Workers (Total 24 Cores) 稼働確認済み*

![vscode](./Images/vscode.png)
*実務を想定し、submissionsやadrを含む洗練されたディレクトリ構造で管理*
## 💡 Key Challenges & Solutions (ADR: 実装の意思決定)

### 1\. ドメイン知識をコードに変換する「物理的特徴量合成」

  - **課題**: 生の反射率データのみでは、木材の個体差や測定時の散乱ノイズに精度が左右される。
  - **解決**: \*\*SNV（標準正規変量変換）\*\*で明るさのムラを消去し、1次微分で波形の「傾き」を抽出。さらに水分吸収帯（1,450nm/1,940nm）の比率や、セルロース・リグニン領域の面積抽出を行い、物理的根拠を持つ特徴量を3,000次元超合成しました。

### 2\. Java 21 世代における「メモリ制限」の突破 (2026-04-08 Update)

  - **課題**: Java 17以降、特に Java 21 では JVM のカプセル化が強化され、Spark/Arrow が内部メモリ（`java.nio`）にアクセスできずクラッシュする。
  - **解決**: `JDK_JAVA_OPTIONS` を Docker レイヤーで定義し、実行パスを `ALL-UNNAMED` に開放。最新世代の Java においても、Spark の分散並列性能を 100% 引き出す基盤を構築しました。

### 3\. ライブラリのメジャーアップデートへの即時適応

  - **課題**: 最新の NumPy 2.x 環境において `np.trapz` が廃止され、実行時エラーが発生。
  - **解決**: APIの変更（`np.trapezoid` への移行）を即座に特定し、最新スタックに準拠した実装へリプレースしました。

### 4\. 提出データの整合性確保（550行問題）

  - **課題**: 提供されたテストデータ（550件）と提出用テンプレート（802件）の不一致。
  - **解決**: `sample_submit.csv` をマスターとした **Left-Join ロジック**を構築。どのような不完全データに対しても100%受理される堅牢な出力エンジンを実装しました。

![challenges](./Images/challenges.png)
*継続的な改善により、着実にスコアを向上させた実績*

### 2. Java 21 世代における「メモリ制限」の突破
- **課題**: Java 21 では JVM のカプセル化が強化され、Spark/Arrow が内部メモリ（`java.nio`）にアクセスできずクラッシュする。
- **解決**: `JDK_JAVA_OPTIONS` を Docker レイヤーで定義し、実行パスを `ALL-UNNAMED` に開放。

![Docker Startup Log](./Images/docker_startup.png)
*JVMオプションが正しく注入され、起動している様子*

## 📁 Directory Structure

```text
.
├── adr/                # 💡 技術選定・意思決定の記録
├── data/
│   ├── raw/            # 📥 Original CSV files
│   ├── processed/      # ⚡ Preprocessed Parquet files
│   └── submissions/    # 📈 Version-controlled submission files
├── src/                # 🐍 Implementation logic
│   ├── preprocess.py   # Spark-based Processing
│   ├── features.py     # Domain Knowledge (SNV, Derivatives)
│   └── main.py         # LightGBM Ensemble & Robust Prediction (Java 21 fix)
├── Dockerfile          # 🐳 Java 21 × Spark 3.5.0 optimized
├── requirements.txt    # 📋 Project dependencies
└── README.md
```

## 📊 Performance & Evolution

  - **Baseline (RF)**: Score 43.87
  - **v3 (LGBM + SNV + 1st Diff)**: Score 42.31
  - **Ultimate (LGBM + Domain FE + Spark Ensemble)**: Score 42.47
  - **Mean CV Accuracy**: **0.9954** (High internal validation performance)

- **Mean CV Accuracy**: **0.9954** (高い内部検証性能を確認)

#### 🔄 Cross Validation Logs
| Fold 0 & 1 | Fold 2 & 3 | Fold 4 (Final) |
| :---: | :---: | :---: |
| ![Fold 0-1](./Images/fold01_log.png) | ![Fold 2-3](./Images/fold23_log.png) | ![Fold 4](./Images/fold4_log.png) |

#### 🏆 Final Score (SIGNATE)
![SIGNATE Score](./Images/signate_score.png)

## 🚀 Getting Started (Docker)

本プロジェクトは Docker を利用して、ローカルに最新の分散処理環境を構築できます。

```bash
# 1. 最強環境のビルド（Java 21 × Spark 3.5.0 専用機）
docker build -t finish-buster-spark .

# 2. パイプラインの実行（24コア並列処理）
docker run --rm -it finish-buster-spark
```
---

### 📝 20260409学習記録　キッチンの配管工事に例えて理解定着を行った

## 🛠 Behind the Scenes: The "Night of Infrastructure"
昨日おこなった作業を、キッチン比喩で解説。

### 1. 最新の耐火基準（Java 21）への適合
昨日は、キッチンの土台を最新の Java 21 に入れ替。
しかし、最新基準はセキュリティが厳しく、**「勝手にガス管（メモリ）に触るな！」**と火を使わせてくれないトラブルが発生。

### 2. 配管のバイパス工事（JVMオプションの設定）
そこで、Dockerfile に特殊な命令を書き込む。
これは、**「このシェフ（Spark）と運搬係（Arrow）に限っては、特別にガス管へのアクセスを許可する」**という特例承認（`--add-opens` オプション）を出す作業でした。これにより、最新の耐火基準を守りつつ、高火力が実現。

### 3. 超高速コンベア（Apache Arrow）の開通
食材運搬係の Arrow が、シェフ（Spark）から盛り付け担当（Pandas）へ、食材を箱に詰め直さず（シリアライズせずに）そのままコンベアで流せるように設定。成功した `toPandas()` の正体。

### 4. Python 3.14 環境における起動プロセスのハック (2026-04-10 Update) 

- **課題**: Python 3.14 (Preview版) および PySpark 4.1.1 環境において、Spark内部のパス解決スクリプト (`find_spark_home.py`) が `pyspark` モジュールを捕捉できず、`AttributeError: 'NoneType' object has no attribute 'origin'` によりセッション起動が停止。
- **解決**: 
  - Windows PowerShell レイヤーで `$env:PYTHONPATH` および `$env:SPARK_HOME` を直接インジェクションし、スクリプトによる自動探索をバイパス。
  - さらに Java 21 の強固なメモリ保護を突破するため、起動オプションに `--add-opens=java.base/java.nio=ALL-UNNAMED` を強制注入。
- **成果**: 最新鋭の実行スタック（Py 3.14 / Java 21 / Spark 4.1.1）での **Spark Session 起動に完全成功**。1,500次元の超多カラムデータに対する Logical Plan (論理計画) の正常性を確認済み。

---
## 🛠 Behind the Scenes: The "Night of Infrastructure" (2026-04-12)
環境構築の試行錯誤を「キッチンの配管工事」に例えて整理。

1. **耐火基準(Java 21)への適合**: セキュリティが厳しく「ガス管(メモリ)に触るな」という警告を、特例承認(`--add-opens`)で突破。
2. **超高速コンベア(Arrow)の開通**: 食材を箱詰め(シリアライズ)せず、そのまま盛り付け担当(Pandas)へ流す高速化を実現。
3. **24口コンロ(24 Cores)の全開**: 当初12コアだった火力を、Workerの増設により24コアまで引き上げ。計算速度を倍増させました。

### 📊 Verification: Data Retrieval Success (2026-04-11) 

インフラの再構築後、1,500次元を超える巨大データフレームから実データを抽出・表示することに成功。最新の実行スタックが理論だけでなく、実務レベルで稼働することを実証しました。

![Data Verification](./Images/success_show.png)
*VSCodeターミナルでの実行結果：`sample number` をキーにスペクトル数値が正しくロードされていることを確認。*

![Spark UI Environment](./Images/spark_env.png)
*Spark UI (Environment): Java 21 (Oracle Corporation) および PySpark 4.1.1 が正常にリンクされ、24スレッドの並列リソースが開放されている状態。*