# NoiseEvaluator Design & Contract

## 1. Overview

画像のノイズ量を、露出・解像度・入力スケール（0..255 / 0..1 / 16bit）に極力依存しない形で推定し、運用用の離散スコア（noise_score）と診断用の生値（sigma系・raw）を同時に返す評価モジュール。raw は他指標との整合のため「高いほど良い」契約に統一し、noise_raw = -noise_sigma_used を固定する。

---

## 2. Output Contract

### 2.1 Output Fields

| Field | Type | Description |
|-------|------|-------------|
| noise_score | float | 5段階離散スコア（1.0 / 0.75 / 0.5 / 0.25 / 0.0）。高いほどノイズが少なく良い。 |
| noise_grade | str | ラベル（excellent / good / fair / poor / bad）。 |
| noise_sigma_midtone | float|null | midtone+低勾配マスク上でのロバストσ（MADベース）。診断用。 |
| noise_sigma_used | float|null | スコア計算に実際に用いたσ（midtoneが使えない場合はglobal_mad等）。小さいほどノイズが少なく良い（物理量寄り）。 |
| noise_mask_ratio | float|null | マスクに採用された画素比率（0..1）。 |
| noise_raw | float|null | 診断・分布調整（score_dist_tune等）用のraw値。契約として「高いほど良い」に統一するため noise_raw = -noise_sigma_used を固定する（sigma_usedが有効な場合）。 |
| noise_eval_status | str | 評価状態（ok / fallback）。 |
| noise_fallback_reason | str|null | フォールバック理由（例: mask_too_small_or_invalid / global_mad_invalid / empty_image / skip など）。 |
| downsampled_size | tuple[int,int]|null | 評価に使った画像サイズ (h, w)。 |
| image_dtype | str|null | 入力画像dtypeの文字列表現（例: uint8 / uint16 / float32）。 |


---

## 3. Purpose

ポートレート評価において「ノイズが少ないほど高評価」を安定して反映する。診断可能性（なぜその評価になったか）を担保しつつ、欠損・異常入力でも破綻しないフォールバックを提供する。さらに、score_dist_tune 等の分布チューニングと連携できるよう raw の向き（高いほど良い）を固定し、将来の再理解コストと逆向きバグを減らす。

---

## 4. Algorithm

1) 入力をグレースケール化し、dtype/スケールを吸収して輝度0..1へ正規化（luma01）。2) 解像度依存を抑えるため長辺downsample。3) GaussianBlurで低周波成分を推定し residual = luma01 - blur を算出。4) midtone（midtone_min..midtone_max）かつ低勾配（Sobelのmagnitude < grad_thr）をマスクとして構成。5) マスク上のresidualからMADベースのロバストσ（noise_sigma_midtone）を算出。6) マスク不足や非有限値の場合はフォールバックとして global MAD（全画素）を用い noise_sigma_used を決定。7) noise_sigma_used を閾値（good_sigma / warn_sigma）に基づき5段階離散スコアへ変換し noise_score / noise_grade を返す。8) 契約として noise_raw = -noise_sigma_used（高いほど良い）を返す。

---

## 5. Normalization Policy

正規化の基本方針は「入力スケール差を評価前に吸収し、σ推定は0..1輝度空間で行う」。具体的には (a) BGR/Gray + uint8/uint16/float を luma01（0..1）へ正規化、(b) downsample_long_edge によって画素数・解像度依存を低減、(c) residual のMADσを採用して外れ値耐性を確保する。スコア化は rawσ（小さいほど良い）を good_sigma / warn_sigma から自動生成した閾値（t1=0.6*g, t2=g, t3=(g+w)/2, t4=w）で 1.0/0.75/0.5/0.25/0.0 に離散化する。分布チューニング用途の raw は「高いほど良い」契約に統一するため noise_raw = -noise_sigma_used とする。

---

## 6. Fallback Policy

マスク割合が min_mask_ratio を下回る、もしくは算出σがNone/NaN/infの場合はフォールバックに遷移し noise_eval_status="fallback" とする。fallback_mode="global_mad" の場合、residual全画素のMADσを noise_sigma_used として採用し、noise_fallback_reason に "mask_too_small_or_invalid" を記録する。global_madでも非有限値となる場合は最終フォールバックとして noise_score=fallback_score（通常0.5）, noise_grade="fair" を返し、reason を "global_mad_invalid" 等で明示する。fallback_mode="skip" 相当の場合は評価を中断せず、フォールバック結果を返して理由を明示する。

---

## 7. Testing

ユニットテストで以下を保証する：1) 基本出力の存在（noise_score/noise_grade/noise_sigma_used/noise_eval_status 等）。2) 離散スコアの妥当性（{0,0.25,0.5,0.75,1.0}に収まる）。3) フォールバック動作（空画像・マスク不足などで必ず安全な出力を返す）。4) 入力異常耐性（ndarray以外で例外、サイズ0でフォールバック）。5) raw契約（noise_raw = -noise_sigma_used）が満たされること（sigma_usedが有効な場合）。必要に応じて uint8 と float01 の入力で過度に崩れないことも監視する。

---

## 8. Future Plans

1) 撮影条件（高ISO/低照度/逆光）別のプロファイル対応。2) 顔領域（face_noise）用に肌テクスチャの影響を抑える追加マスク（肌色/周波数帯）を検討。3) バッチ分布（score_dist_tune等）と連携した閾値自動調整（まずは good_sigma/warn_sigma のチューニング補助）。4) suggestions（自動提案）は原則“自動適用しない”：ok行優先・偏り検知・変更量制限・PR差分でレビュー可能、を運用ルールとして固定。5) 露出スコアに基づくノイズ補正（brightness compensation）を設計上の拡張点として整理。

---

## 9. Design Philosophy

Raw（診断）とScore（運用）を分離し、スコアは比較・合成に強い離散値へ正規化する。異常入力・条件不足を黙殺せず、必ず eval_status と fallback_reason で理由を残す。「落ちない」「追える」「揺れにくい」を優先する。raw の向きは全指標で統一（高いほど良い）し、ノイズは符号反転（noise_raw = -noise_sigma_used）で揃える。score_dist_tune の suggestions は便利だが“提案”であり、運用ルールに従って段階的に適用する。

---

## 10. Change History

| Date | Change | Author |
|------|--------|--------|
| 2026-02 | MADベースσ推定（midtone+low_gradマスク）と、mask不足時のglobal_madフォールバック、5段階離散スコア化、診断用フィールド（sigma/mask_ratio/status/reason）を整備 | akky0108 |
| 2026-02 | CSV互換・分布チューニング連携のため noise_raw を追加し、raw契約（高いほど良い）に統一：noise_raw = -noise_sigma_used を固定（逆向きバグ防止、score_dist_tune前提の明確化） | akky0108 |
