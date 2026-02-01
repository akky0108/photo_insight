# Evaluation Rank CSV Contract

このドキュメントは  
`evaluation_results_*.csv`（入力）から  
`evaluation_ranking_*.csv`（出力）を生成する  
**EvaluationRankBatchProcessor 一式の I/O 契約（Contract）**を定義する。

本契約は以下を目的とする：

- 評価・採用・Lightroom 付与処理における 列名・順序の固定
- 将来の指標追加・削除を 破壊的変更にしない
- GitHub 上での PR レビュー基準の明確化
- CSV を中間 API とした長期運用の安定化

---

## 1. スコープ

対象モジュール：

- `evaluation_rank_batch_processor.py`
- `evaluation_rank/scoring.py`
- `evaluation_rank/acceptance.py`
- `evaluation_rank/lightroom.py`
- `evaluation_rank/writer.py`

本契約は CSV の列構造・順序のみを対象とする。
以下は対象外とする：

    - スコア算出ロジック
    - 閾値・重みの妥当性
    - 評価アルゴリズムの内部実装

---

## 2. 入力CSV（evaluation_results_*.csv）

### 2.1 入力ヘッダ（完全定義）

```csv
file_name,
sharpness_score,sharpness_raw,sharpness_eval_status,
blurriness_score,blurriness_raw,blurriness_grade,blurriness_eval_status,blurriness_fallback_reason,
contrast_score,contrast_raw,contrast_eval_status,contrast_grade,
noise_score,noise_grade,noise_sigma_midtone,noise_sigma_used,noise_mask_ratio,noise_eval_status,noise_fallback_reason,
local_sharpness_score,local_sharpness_raw,local_sharpness_std,local_sharpness_eval_status,local_sharpness_fallback_reason,
local_contrast_score,local_contrast_raw,local_contrast_std,local_contrast_eval_status,local_contrast_fallback_reason,
exposure_score,mean_brightness,exposure_grade,exposure_eval_status,exposure_fallback_reason,
blurriness_score_brightness_adjusted,noise_score_brightness_adjusted,
full_body_detected,pose_score,headroom_ratio,footroom_ratio,side_margin_min_ratio,full_body_cut_risk,
body_height_ratio,body_center_y_ratio,
face_detected,faces,
face_sharpness_score,face_sharpness_raw,face_sharpness_eval_status,
face_contrast_score,face_contrast_raw,face_contrast_eval_status,face_contrast_grade,
face_noise_score,face_noise_grade,face_noise_sigma_midtone,face_noise_mask_ratio,
face_local_sharpness_score,face_local_sharpness_std,
face_local_contrast_score,face_local_contrast_std,
face_exposure_score,face_mean_brightness,face_exposure_grade,face_exposure_eval_status,face_exposure_fallback_reason,
yaw,pitch,roll,gaze,
delta_face_sharpness,delta_face_contrast,
face_composition_raw,face_composition_score,face_composition_status,
face_blurriness_raw,face_blurriness_score,face_blurriness_grade,face_blurriness_eval_status,face_blurriness_fallback_reason,
face_blurriness_score_brightness_adjusted,
expression_score,expression_grade,
composition_rule_based_score,face_position_score,framing_score,face_direction_score,eye_contact_score,
lead_room_score,
body_composition_raw,body_composition_score,
composition_raw,composition_score,composition_status,
main_subject_center_source,main_subject_center_x,main_subject_center_y,
rule_of_thirds_raw,rule_of_thirds_score,
contrib_comp_composition_rule_based_score,contrib_comp_face_position_score,contrib_comp_framing_score,
contrib_comp_lead_room_score,contrib_comp_body_composition_score,contrib_comp_rule_of_thirds_score,
group_id,subgroup_id,shot_type,
accepted_flag,accepted_reason
```

### 2.2 入力CSVの責務

- evaluators 層の最終成果物
- ranking 処理は入力列を破壊・上書きしない
- faces は JSON list 形式を想定
- 顔選択・統合処理は ranking 側の責務とする

## 3. 出力CSV（evaluation_ranking_*.csv）
### 3.1 出力ヘッダ（完全定義）
```csv
file_name,group_id,subgroup_id,shot_type,
face_detected,category,
overall_score,
flag,accepted_flag,secondary_accept_flag,

blurriness_score,sharpness_score,contrast_score,noise_score,
local_sharpness_score,local_contrast_score,

face_sharpness_score,face_contrast_score,face_noise_score,
face_local_sharpness_score,face_local_contrast_score,

composition_rule_based_score,face_position_score,framing_score,
face_direction_score,eye_contact_score,

debug_pitch,debug_gaze_y,debug_eye_contact,
debug_expression,debug_half_penalty,debug_expr_effective,

score_composition,score_face,score_technical,

contrib_comp_body_composition_score,
contrib_comp_composition_rule_based_score,
contrib_comp_eye_contact_score,
contrib_comp_face_direction_score,
contrib_comp_face_position_score,
contrib_comp_framing_score,
contrib_comp_lead_room_score,
contrib_comp_rule_of_thirds_score,

contrib_face_contrast,
contrib_face_exposure,
contrib_face_expression,
contrib_face_local_contrast,
contrib_face_local_sharpness,
contrib_face_noise,
contrib_face_sharpness,

contrib_tech_blurriness,
contrib_tech_exposure,
contrib_tech_local_sharpness,
contrib_tech_noise,
contrib_tech_sharpness,

lr_keywords,lr_rating,lr_color_label,lr_labelcolor_key,lr_label_display,
accepted_reason
```

### 3.2 出力CSVの責務

- ranking / acceptance / lightroom 処理の最終成果物
- Lightroom 実運用を前提とした可視性重視設計
- lr_* 系は Single Source of Truth
- accepted_reason は Green / Yellow 共通化

## 4. 列の分類（設計ルール）
### 4.1 必須入力列（Contract Required）

file_name
face_detected
group_id
subgroup_id
shot_type
各 *_score 系（technical / face / composition）

### 4.2 ranking 生成列

category
score_technical
score_face
score_composition
overall_score

### 4.3 acceptance 専用列

accepted_flag
secondary_accept_flag
flag
accepted_reason

### 4.4 debug 列

debug_* は 自由追加可
ranking ロジックに依存してはならない

## 5 Acceptance（Green / Yellow 配分ルール）

acceptance.py における最終採用（Green）は、
グループ内件数に応じて動的に比率が決定される。

### Green 配分ルール（デフォルト）

| 件数 n | 比率 | 備考 |
|--------|------|------|
| n ≤ 60 | 30% | 少数精鋭優先 |
| 61–120 | 25% | 中規模安定帯 |
| n >120 | 20% | 大量時の品質維持 |

算出式：

green_total = max(ceil(n * ratio), green_min_total)

### Green per-group quota（偏り防止）

green_per_group_enabled が有効な場合、Green（accepted_flag=1）は accept_group ごとに
一定の枠（quota）を優先的に確保し、特定グループへの偏りを抑制する。

- accept_group ごとに quota を算出し、まず各グループ内の上位から埋める
- ただし、グループ側で gate を満たせず quota を満たせない場合がある
  - その不足分は「全体 backfill」に回し、green_total を必ず充足する（枠欠損を起こさない）
- green_per_group_min_each は「グループ最低保証」の意図だが、全体枠が小さすぎる場合は
  “強いグループ優先” の配分となり、実質 0 枠のグループが発生しうる


### Backfill ポリシー

Green 枠（green_total）は必ず充足される設計とする。
選定は overall_score 上位順を基本に、以下の **3段階**で backfill を行う：

1. strict gate（内容フィルタ）  
2. relaxed gate（不足時の救済：内容条件を一段緩和）  
3. forced gate（最終補完：内容条件は大きく緩和するが、半目NGなどのポリシーは維持）

この backfill により、特定グループの quota が内容 gate で埋められない場合でも、
不足分は全体 backfill により補完され、green_total の欠損は発生しない。


#### accepted_reason の backfill 表示

Green に採用された行の accepted_reason には、運用・デバッグ可視化のため
backfill 由来を示すサフィックスを付与してよい。

- strict gate 由来：サフィックス無し
- relaxed gate 由来：FILL_RELAX
- forced gate 由来：FILL_FORCED


### Eye state policy（半目/閉眼）

portrait かつ face_detected の場合、目状態推定に基づく最終ポリシーを適用する。

- half（半目）: accepted_flag / secondary_accept_flag を強制 0（採用不可）
- closed（閉眼）: warn（採用は落とさない。注意喚起は accepted_reason 等に残してよい）
- 推定の信頼性が不足する場合（eye_patch_size が閾値未満など）は unknown とし、判定しない


### ログ用補助情報

apply_accepted_flags() の戻り値には以下を含む：

- green_ratio_effective
- total_n
- green_per_group_enabled


これらは運用・分析・デバッグ用途に限定して使用する。
CSV の列構造・順序（Contract）には影響しない。


> NOTE: 本章の変更は acceptance の内部挙動（配分/補完/理由表示）に関するものであり、
> 入出力CSVの列名・順序には影響しない。


## 6. 変更ルール（GitHub 運用）

- 許可される変更
    debug_* 列の追加
    contrib_* 列の追加
    accepted_reason の文言変更
    lr_keywords の表記調整

- 破壊的変更（要 PR 明示）
    列名変更
    列削除
    列順変更
    必須列の削除
    型変更（bool → int など）

    これらは必ず PR 内で明示し、影響範囲を記載すること。

- 型の変更（bool → 数値など）

## 7. この契約の位置付け

- 本ドキュメントはコードより優先される
- 実装は本契約に従属する
- PR レビューでは本契約との整合性を最優先で確認する

CSV は中間 API として扱う。

## 8. 自動検証（pytest / CI）

本契約は以下により自動検証される：
    - test_contract_headers.py
    - test_lightroom.py
    - GitHub Actions CI

検証内容：
    - 列重複チェック
    - 列順固定
    - 欠損検知
    - 実CSV照合

## 9. 次のステップ（別PR）

- 入力CSV contract validation の強化
- 出力CSV品質テスト（分布・比率）
- acceptance ルール分割
- 閾値のYAML化
- 分析スクリプト導入
