## 📸 Photo Insight

画像評価処理を支援するツール群です。評価指標のスコア計算やログ出力、バッチ処理などの機能を備えています。

## 📁 プロジェクト構成

photo_insight/
├── src/ # ソースコード
│ └── ...
├── config/ # 設定ファイル（例: ログ設定）
│ └── logging_config.yaml
├── utils/ # ユーティリティ（AppLoggerなど）
│ └── app_logger.py
└── README.md # このドキュメント

## 🧪 環境構築
- Python: **3.10**
- 仮想環境名: **photo_eval_env**

```bash
# conda 環境の作成
conda create -n photo_eval_env python=3.10
conda activate photo_eval_env

# 依存パッケージのインストール（必要に応じて）
pip install -r src/photo_eval_env_manager/requirements.txt
```

## 📝 ログ機能

`utils.app_logger.AppLogger` を使用することで、プロジェクトルートにログファイルを出力するカスタムロガーを簡単に利用できます。

```python
from utils.app_logger import AppLogger

logger = AppLogger(project_root=".", logger_name="MyLogger").get_logger()
logger.info("ログ出力テスト")
```

出力先（例）: ./logs/MyLogger/app.log

## 🧪 ノートブックによる利用例

`notebooks/app_logger_example.ipynb` にて、ログ出力機能の実行例を確認できます。


## 📄 ライセンス
MIT License