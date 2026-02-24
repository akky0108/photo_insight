## Batch Framework

本プロジェクトのバッチ処理は、BaseBatchProcessor を中心とした 統一ライフサイクル設計 に基づいて実装されています。

目的は以下です。
- データ取得の 一回化・再現性の担保
- 副作用の分離による 保守性向上
- 並列処理・再開処理・エラーハンドリングの 共通化
- テストで契約を保証できる 運用基盤の構築

### ライフサイクル概要

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

### 新契約（重要）
1. load_data()（必須）

- 純I/O専用
- CSV読み込み、ファイル列挙、DB/HTTP取得など
- 副作用（self.xxx への代入・統計計算・初期化）は 禁止

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

### 実装テンプレート（推奨）
```python
from typing import Dict, List, Any, Optional
from photo_insight.core.batch_framework.base_batch import BaseBatchProcessor

class SampleBatchProcessor(BaseBatchProcessor):
    def __init__(self, config_path: Optional[str] = None, max_workers: int = 2, logger=None):
        super().__init__(config_path=config_path, max_workers=max_workers, logger=logger)
        self.paths: Dict[str, str] = {}

    def setup(self) -> None:
        self.paths.setdefault("input_dir", "./temp")
        self.paths.setdefault("output_dir", "./output")
        super().setup()

    def load_data(self) -> List[Dict[str, Any]]:
        return [{"x": 1}, {"x": 2}]

    def after_data_loaded(self, data: List[Dict]) -> None:
        self.logger.info(f"Loaded {len(data)} items")

    def _process_batch(self, batch: List[Dict]) -> List[Dict[str, Any]]:
        return [{"status": "success", "row": item} for item in batch]

    def cleanup(self) -> None:
        super().cleanup()
```

### 実装ルール（運用指針）

- execute() を サブクラスで override しない
- 再開・残数・運用ログは after_data_loaded() / cleanup() に集約
- 並列書き込みがある場合は get_lock() を使用
- 出力処理は可能な限り writer / exporter に分離する

### 代表的なバッチ実装例
#### EvaluationRank

- load_data()：評価CSV読み込み
- after_data_loaded()：P95 calibration 構築
- cleanup()：ランキング・accepted判定・CSV出力

#### NEFFileBatchProcess

- setup()：対象ディレクトリ列挙
- load_data()：NEFファイル一覧収集
- _process_batch()：EXIF抽出＋CSV追記（ロック付き）

#### PortraitQualityBatchProcessor

- setup()：パス設定・既処理データ読み込み
- load_data()：未処理画像のみ抽出
- after_data_loaded()：再開・件数・閾値ログ
- _process_batch()：評価・保存・メモリ監視

### テストと契約保証

以下のテストにより、Batch Framework の契約は保証されています。

- #### Data契約
    - setup() で load_data() が1回だけ呼ばれる
    - process() で再ロードされない

- #### 例外契約
    - フック例外はログに残る
    - cleanup() 例外は最終的に RuntimeError

（tests/unit/processors/ 参照）

## Portrait Quality Evaluation – Design Memo
### 目的

ポートレート写真を
- 技術品質
- 顔品質
- 構図
- 全身構図（full body）
の複数軸で評価し、
運用可能な accepted / rejected 判定と理由を安定して出力する。

### 設計方針：

- 閾値ロジックの明確化
- CSV出力の安定性
- Lightroom / 再学習 / 集計への拡張性

### テストによる仕様固定


評価パイプライン概要
```text
画像
 ↓
PortraitQualityEvaluator.evaluate()
 ├─ 顔検出（insightface）
 ├─ 全身検出（MediaPipe Pose）
 ├─ 技術評価（sharpness / noise / contrast / blur / exposure）
 ├─ 顔領域評価（face_*）
 ├─ 構図評価（rule-based）
 ├─ 派生指標
 │   ├─ lead_room_score
 │   ├─ delta_face_sharpness
 │   └─ delta_face_contrast
 └─ accepted 判定（4ルート）
 ↓
PortraitQualityBatchProcessor
 ↓
CSV 出力（ヘッダ保証済み）
```

### accepted 判定ロジック

判定は 4ルート制（優先順位あり）

```text
 face_quality
 > composition
 > technical
 > full_body
```

#### Route C: face_quality（最優先）

「顔が良ければ採用」

- 顔が検出されている
- 顔のシャープネス・ノイズが高い
- 顔が不自然に悪化していない（delta_face_*）
- yaw が大きすぎない

```python
accepted_reason = "face_quality"
```

#### Route A: composition

「構図が良ければ採用」

- 三分割・フレーミングが良好
- 視線方向に余白がある（lead_room_score）
- 最低限の技術品質を満たす

```python
accepted_reason = "composition"
```

#### Route B: technical

「技術的に非常に安定していれば採用」

- ノイズ・コントラスト・ブレが高水準
- 顔品質が致命的に悪化していない

```python
accepted_reason = "technical"
```

#### Route D: full_body（顔なし救済ルート）

「顔が無くても全身構図として成立していれば採用」

- full_body_detected == True
- pose の視認性が十分（pose_score）
- 切れリスクが低い（full_body_cut_risk）
- 最低限の技術品質を満たす

```python
accepted_reason = "full_body"
```

失敗時：
```python
accepted_reason = "full_body_rejected"
```

#### Rejected

どのルートにも該当しない場合：

```python
accepted_flag = False
accepted_reason = "rejected"
```

### 出力フィールド設計（CSV）
#### 顔系
- face_detected
- face_*_score
- yaw / pitch / roll / gaze
- delta_face_sharpness
- delta_face_contrast

#### 画像全体
- sharpness_score
- noise_score
- contrast_score
- blurriness_score
- exposure_score

#### 構図
- composition_rule_based_score
- framing_score
- face_position_score
- face_direction_score
- lead_room_score

#### 全身
- full_body_detected
- pose_score
- headroom_ratio
- footroom_ratio
- side_margin_min_ratio
- full_body_cut_risk

#### 判定メタ
- accepted_flag
- accepted_reason

※ ヘッダは PortraitQualityHeaderGenerator で一元管理
※ CSV 出力はテストで保証済み

### テスト戦略（要点）
レイヤ	目的
Evaluator	accepted 分岐・優先順位を固定
Header	CSV ヘッダ漏れ防止
Batch	save_results が実際に書くことを保証

#### 重要
評価ロジックを変える場合は、

テストが落ちる → 仕様を決める → 通す

の順で必ず行うこと。

### 次にやるなら（メモ）
- full_body 用の構図スコア（立ち姿勢・重心）
- pose_bbox の CSV 分解（x/y/w/h）
- accepted_reason の Lightroom ラベル連携
- 将来の ML 学習用 feature freeze