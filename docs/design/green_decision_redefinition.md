# Green判定の再定義

- Issue: #317 [2026-0030] Green判定の再定義（最低成立＋残す理由）
- Status: Implemented
- Updated: 2026-04-14

## 1. 目的

Green判定を、単なる閾値通過ではなく、次の意味に再定義する。

**Green = 最低成立を満たし、残す理由が1つ以上ある**

これにより、Green の意味を「成立していて、残す根拠がある写真」に揃える。

---

## 2. 背景

従来の Green 判定では、次の問題が起きうる。

- Green だが残す理由が弱い写真が混ざる
- ピンボケや成立不足が Green に入る余地がある
- なぜ Green になったか説明しづらい
- 閾値調整や見直しのときに根拠を追いにくい

このため、Green を単一の総合点や既存 accepted 系フラグと同一視せず、独立した品質判定として定義し直した。

---

## 3. 方針

Green 判定は2段階で行う。

### 3.1 最低成立
候補として残してよい最低ラインを満たしていること。

実装では、実データを踏まえて以下を最低成立に採用した。

- `shot_type != "no_face"`
- `face_detected == True`
- `blurriness_score >= min_blurriness_score`
- `composition_score >= min_composition_score`
- 顔が検出されている場合は `face_exposure_score >= min_face_exposure_score`

### 3.2 残す理由
その写真を積極的に残す理由があること。

実装では、以下の3軸で keep reason を収集する。

- SNS受け
- モデル受け
- プロ品質

---

## 4. 判定ルール

最終判定は以下とする。

```text
is_green = minimum_pass and keep_reasons が1件以上
```

### 4.1 最低成立NG
以下のどれかを満たさない場合は Green にしない。

- `shot_type == "no_face"`
- `face_detected == False`
- `blurriness_score < min_blurriness_score`
- `composition_score < min_composition_score`
- 顔あり画像で `face_exposure_score < min_face_exposure_score`

### 4.2 最低成立OK
最低成立を満たした場合のみ、残す理由を集める。

### 4.3 残す理由あり
残す理由が1件以上あれば Green とする。

### 4.4 残す理由なし
最低成立を満たしていても、残す理由がなければ Green にしない。

---

## 5. データ構造

```python
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class GreenDecision:
    is_green: bool
    minimum_pass: bool
    keep_reasons: List[str]
    reject_reasons: List[str]
```

### 項目の意味
- `is_green`: 最終 Green 判定
- `minimum_pass`: 最低成立を通ったか
- `keep_reasons`: 残す理由
- `reject_reasons`: Green にしなかった理由

---

## 6. 実装方針

Green 判定は専用関数に切り出した。

実装配置:

```text
src/photo_insight/pipelines/evaluation_rank/_internal/green_decision.py
```

実装した主な関数:

```python
def is_green_minimum_pass(row: dict, thresholds: dict) -> tuple[bool, list[str]]:
    ...

def collect_green_keep_reasons(row: dict, thresholds: dict) -> list[str]:
    ...

def decide_green(row: dict, thresholds: dict) -> GreenDecision:
    ...

def apply_green_decision_to_row(row: dict, thresholds: dict) -> dict[str, Any]:
    ...
```

### 実装上の役割分担
- `is_green_minimum_pass`
  - 最低成立判定と reject reason の生成
- `collect_green_keep_reasons`
  - keep reason の収集
- `decide_green`
  - 最終判定の組み立て
- `apply_green_decision_to_row`
  - CSV出力向けの列セットを返す

---

## 7. 閾値管理

閾値は `green_decision` 設定から読み込める構成とした。

```yaml
green_decision:
  version: "v2-real"
  require_face_detected: true
  min_blurriness_score: 0.50
  min_composition_score: 0.50
  min_face_exposure_score: 0.50
  reject_no_face_shot_type: true

  keep_reason_thresholds:
    expression_score: 0.75
    composition_score: 0.75
    eye_contact_score: 0.75
    face_position_score: 0.75
    face_exposure_score: 0.75
    face_sharpness_score: 0.75
    pose_score: 75.0
    pro_composition_strong: 0.85
    pro_face_composition_strong: 0.85
```

### 実装上の初期値
- `version = "v2-real"`
- `min_blurriness_score = 0.50`
- `min_composition_score = 0.50`
- `min_face_exposure_score = 0.50`

※ 数値は初期運用値であり、実データ確認後に微調整可能とする。

---

## 8. keep reason の定義

### 8.1 SNS受け
以下を keep reason として採用する。

- `sns_expression`
- `sns_composition`
- `sns_eye_contact`
- `sns_face_position`

判定に使う列:
- `expression_score`
- `composition_score`
- `eye_contact_score`
- `face_position_score`

### 8.2 モデル受け
以下を keep reason として採用する。

- `model_face_exposure`
- `model_face_sharpness`
- `model_pose`

判定に使う列:
- `face_exposure_score`
- `face_sharpness_score`
- `pose_score`

### 8.3 プロ品質
以下を keep reason として採用する。

- `pro_composition_strong`
- `pro_face_composition_strong`

判定に使う列:
- `composition_score`
- `face_composition_score`

---

## 9. 出力項目

今回の実装で、ranking CSV に以下を追加した。

- `is_green`
- `green_minimum_pass`
- `green_keep_reasons`
- `green_reject_reasons`
- `green_decision_version`

出力例:

```text
is_green = 1
green_minimum_pass = 1
green_keep_reasons = sns_expression|model_face_exposure
green_reject_reasons =
green_decision_version = v2-real
```

最低成立NGの例:

```text
is_green = 0
green_minimum_pass = 0
green_keep_reasons =
green_reject_reasons = blurriness_too_low
green_decision_version = v2-real
```

### writer / contract 反映
- `contract.py`
  - `OUTPUT_COLUMNS` に Green 系列を追加
- `writer.py`
  - `is_green`
  - `green_minimum_pass`

  を 0/1 正規化対象に追加

---

## 10. 本体組み込み

`EvaluationRankBatchProcessor` に Green 判定を組み込んだ。

組み込み位置:
- `_process_batch()` 内
- `overall_score` / `score_*` / `contrib_*` 計算後
- `results.append(...)` の直前

処理イメージ:

```python
green_updates = apply_green_decision_to_row(
    out,
    (self.config_manager.get_config() or {}),
)
out.update(green_updates)
```

これにより、evaluation_rank 本体で算出された行ごとに Green 判定結果が付与される。

---

## 11. テスト方針

以下を単体テストで確認した。

1. `no_face` は minimum pass で落ちる
2. `face_detected=False` なら Green にならない
3. 最低成立OKでも理由なしなら Green にならない
4. SNS理由で Green になる
5. モデル受け理由で Green になる
6. 複数理由を保持できる
7. custom threshold が反映される
8. writer が Green 系列を出力できる

想定ファイル:

```text
tests/unit/pipelines/evaluation_rank/test_green_decision.py
tests/unit/pipelines/evaluation_rank/test_writer_green_columns.py
```

### 実施結果
- `tests/unit/pipelines/evaluation_rank/test_green_decision.py` 通過
- `tests/unit/pipelines/evaluation_rank/test_writer_green_columns.py` 通過
- `tests/unit/pipelines/evaluation_rank/` 全体通過

---

## 12. 実データ確認結果

実データのCSV列構成を確認した上で、Green 判定に使う列を選定した。

主に使用した列:
- `shot_type`
- `face_detected`
- `blurriness_score`
- `composition_score`
- `face_exposure_score`
- `expression_score`
- `eye_contact_score`
- `face_position_score`
- `face_sharpness_score`
- `pose_score`
- `face_composition_score`

確認した内容:
- Green 系列が ranking CSV に出力される
- `green_reject_reasons` によって落ちた理由が説明可能
- `secondary_accept_flag` と `is_green` が独立している
- 既存 accepted 系との衝突はない

---

## 13. 完了条件

今回の対応で、以下を満たした。

- Green 判定が「最低成立＋残す理由」に置き換わっている
- 判定理由をコード上で説明できる
- 単体テストが追加されている
- ranking CSV に理由が出力される
- `evaluation_rank` 配下の unit test が全通している

---

## 14. 今回の非目標

今回の対応では以下は直接扱わない。

- portrait / body / non_face の分類拡張
- 分布健全性チェックそのもの
- Lightroom UI 全体の見直し
- ユーザー主観選定の完全再現
- accepted_flag と Green の統合

---

## 15. 今後の論点

今後の調整候補は以下。

- `min_blurriness_score` の厳しさ調整
- `min_composition_score` の厳しさ調整
- SNS受け・モデル受けの精度向上
- keep reason を Lightroom keyword へ反映するかの検討
- Green と accepted の関係再整理

---

## 16. まとめ

今回の変更で、Green は次の意味になった。

**Green = 最低成立を満たし、残す理由がある写真**

さらに実装上は、以下の3軸で理由を収集する。

- SNS受け
- モデル受け
- プロ品質

これにより、Green の意味を明確化し、判定理由を追えるようにした。
また、後からの見直しや閾値調整がしやすい構造になった。
