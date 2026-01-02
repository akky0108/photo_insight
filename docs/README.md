## Batch Framework

本プロジェクトのバッチ処理は、BaseBatchProcessor を中心とした 統一ライフサイクル設計に基づいて実装されています。
目的は以下です。

- データ取得の 一回化・再現性の担保
- 副作用の分離による 保守性向上
- 並列処理・再開処理・エラーハンドリングの 共通化
- テストで契約を保証できる基盤の構築

## ライフサイクル概要

すべてのバッチは processor.execute() をエントリポイントとして実行されます。

```powershell
execute()
 ├─ setup()
 │   ├─ load_data()          # データ取得（純I/O）
 │   └─ after_data_loaded()  # 副作用（1回だけ）
 ├─ process()
 │   └─ _process_batch()     # バッチ単位の処理（並列）
 └─ cleanup()
```

## 新契約（重要）
1. load_data()（必須）

- 純I/O専用
- CSV読み込み、ファイル列挙、DB/HTTP取得など
- 副作用（self.xxx への代入・統計計算・初期化）は禁止

```python
def load_data(self) -> List[Dict]:
    return read_csv(...)
```

2. after_data_loaded(data)（任意）

- load_data() の結果に基づく 副作用を1回だけ実行
- calibration 構築、閾値計算、初期化ログなど

```python
def after_data_loaded(self, data: List[Dict]) -> None:
    self.calibration = build_calibration(data)
```

3. get_data() は override しない

- get_data() は BaseBatchProcessor が保持
- キャッシュ・再利用・二重呼び出し防止を担う
- サブクラスで override すると 警告ログが出る（非推奨）

## 実装テンプレート（推奨）
```python
from typing import Dict, List, Any, Optional
from batch_framework.base_batch import BaseBatchProcessor

class SampleBatchProcessor(BaseBatchProcessor):
    def __init__(self, config_path: Optional[str] = None, max_workers: int = 2, logger=None):
        super().__init__(config_path=config_path, max_workers=max_workers, logger=logger)
        self.paths: Dict[str, str] = {}

    def setup(self) -> None:
        # load_data() 実行前に必要な準備のみ行う
        self.paths.setdefault("input_dir", "./temp")
        self.paths.setdefault("output_dir", "./output")
        super().setup()

    def load_data(self) -> List[Dict[str, Any]]:
        # 純I/Oのみ
        return [{"x": 1}, {"x": 2}]

    def after_data_loaded(self, data: List[Dict]) -> None:
        # 副作用はここに集約
        self.logger.info(f"Loaded {len(data)} items")

    def _process_batch(self, batch: List[Dict]) -> List[Dict[str, Any]]:
        results = []
        for item in batch:
            results.append({"status": "success", "score": 1.0, "row": item})
        return results

    def cleanup(self) -> None:
        super().cleanup()
```

## 実装ルール（運用指針）

- execute() をサブクラスで override しない
- エラーハンドリング、フック、統計集約の恩恵を維持するため
- 再開・残数・運用ログは after_data_loaded() や cleanup() に寄せる
- 並列書き込みがある場合は get_lock() を使用する
- 出力処理は可能な限り writer モジュール等に分離する

## 代表的なバッチ実装例

- EvaluationRank

    - load_data()：評価CSV読み込み
    - after_data_loaded()：P95 calibration 構築
    - cleanup()：ランキング・accepted判定・CSV出力

- NEFFileBatchProcess
    - setup()：対象ディレクトリ列挙
    - load_data()：NEFファイル一覧収集
    - _process_batch()：EXIF抽出＋CSV追記（ロック付き）

- PortraitQualityBatchProcessor
    - setup()：パス設定・既処理データ読み込み
    - load_data()：未処理画像のみ抽出
    - after_data_loaded()：再開・件数・閾値ログ
    - _process_batch()：評価・保存・メモリ監視

## テストと契約保証

以下のテストにより、Batch Framework の契約は自動的に保証されています。

- Data契約
    - setup() で load_data() が1回だけ呼ばれる
    - process() で再ロードされない

- 例外契約
    - フック例外はログに残る
    - cleanup 例外は最終的に RuntimeError となる

（tests/unit/processors/ 参照）