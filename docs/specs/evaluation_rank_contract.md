# Evaluation Rank CSV Contract

このドキュメントは  
`evaluation_results_*.csv`（入力）から  
`evaluation_ranking_*.csv`（出力）を生成する  
**EvaluationRankBatchProcessor 一式の I/O 契約（Contract）**を定義する。

本契約は以下を目的とする：

- 評価・採用・Lightroom 付与処理の **列名整合性の固定**
- 将来の指標追加・削除を **破壊的変更にしない**
- GitHub 上での PR レビュー基準を明確化する

---

## 1. スコープ

対象モジュール：

- `evaluation_rank_batch_processor.py`
- `evaluation_rank/scoring.py`
- `evaluation_rank/acceptance.py`
- `evaluation_rank/lightroom.py`
- `evaluation_rank/writer.py`

本契約は **CSV の列構造のみ**を扱い、  
スコア計算ロジックや閾値の是非は含まない。

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

評価器（evaluators）層の最終成果物

ranking 処理は 入力列を破壊・上書きしない

faces は JSON list を想定（best face 抽出は ranking 側）

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

- ranking / acceptance / lightroom 処理の 最終成果物
- Lightroom での実運用を前提とした 可視性優先
- accepted_reason は Green / Yellow 共通で1本化

## 4. 列の分類（設計ルール）
### 4.1 必須入力列（Contract Required）

file_name
face_detected
group_id
subgroup_id
shot_type
各 *_score 系（tech / face / composition）

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

## 5. 変更ルール（GitHub 運用）

- 許可される変更
    debug 列の追加
    contrib_* 列の追加
    accepted_reason の文言変更
- 破壊的変更（要 PR 明示）
    列名変更
    必須列の削除

- 型の変更（bool → 数値など）

## 6. この契約の位置付け

本 md は コードより優先される
実装は本契約に 従属する
PR レビュー時は「この契約を壊していないか」を最初に確認する

## 7. 次のステップ（別PR）

入力CSVの contract validation（列チェック）
出力CSVの contract validation（列順・欠損チェック）
pytest による自動検証
