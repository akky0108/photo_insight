## 📸 Photo Insight

画像評価処理を支援するツール群です。評価指標のスコア計算やログの出力、バッチ処理などを備えています。

## 📁 プロジェクト構成

photo_insight/
├── src/
│ └── ...
├── config/
│ └── logging_config.yaml
├── utils/
│ └── app_logger.py
└── README.md

## 🧪 環境構築

- Python: 3.10
- 仮想環境名: `photo_eval_env`

```bash
# conda 環境の作成（例）
conda create -n photo_eval_env python=3.10
```

## 📝 ログユーティリティの使い方

```python
from utils.app_logger import AppLogger

logger = AppLogger(project_root=".", logger_name="MyLogger").get_logger()
logger.info("Hello from logger!")
```

## 📄 ライセンス
MIT License