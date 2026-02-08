# BlurrinessEvaluator Design & Contract

## 1. Overview

画像の「ぼやけ（=シャープさ）」を、入力の dtype/スケール差に極力依存しない形で推定し、運用用の5段階離散スコア（blurriness_score）と診断用の生値（blurriness_raw / variance_of_*）を同時に返す評価モジュール。raw はコントラクトとして「高いほど良い（higher_is_better）」に統一し、常に blurriness_raw_direction / blurriness_raw_transform / blurriness_higher_is_better を欠損なく出力する。

---

## 2. Output Contract

### 2.1 Output Fields

| Field | Type | Description |
|-------|------|-------------|
| blurriness_score | float | 5段階離散スコア（1.0 / 0.75 / 0.5 / 0.25 / 0.0）。高いほどシャープで良い。 |
| blurriness_grade | str | ラベル（excellent / good / fair / blurry / very_blurry）。 |
| blurriness_raw | float | 診断用の統合raw値（分散特徴の重み付き和）。契約として「高いほど良い（higher_is_better）」を固定する。 |
| blurriness_raw_direction | str | rawの向き。常に "higher_is_better"。 |
| blurriness_raw_transform | str | rawの変換。常に "identity"。 |
| blurriness_higher_is_better | bool | rawの向きのbool表現。常に true。 |
| blurriness_eval_status | str | 評価状態（ok / fallback / invalid）。 |
| blurriness_fallback_reason | str|null | fallback/invalid の理由（ok は空文字）。例: invalid_input_not_ndarray / invalid_input_empty / too_small_image_{w}x{h} / non_finite_raw / exception:TypeName |
| variance_of_gradient | float | Sobel勾配強度（magnitude）の分散。診断用。 |
| variance_of_laplacian | float | Laplacianの分散。診断用。 |
| variance_of_difference | float | GaussianBlurとの差分（absdiff）の分散。診断用。 |


---

## 3. Purpose

ポートレート評価において「ピントが合っている（シャープ）ほど高評価」を安定して反映する。診断可能性（なぜその評価になったか）を担保しつつ、欠損・異常入力でも破綻しないフォールバックを提供する。さらに、分布チューニング（score_dist_tune）で得た閾値を config の SSOT として管理し、運用中の挙動が再現可能になるようにする。

---

## 4. Algorithm

1) 入力をグレースケール化し、dtype/スケールを吸収して float32[0..1] へ正規化（gray01）。2) Sobel（Tenengrad）から勾配強度を計算し、その分散（variance_of_gradient）を得る。3) Laplacian を計算し、その分散（variance_of_laplacian）を得る。4) GaussianBlur による低周波推定と差分（absdiff）を取り、その分散（variance_of_difference）を得る。5) 3つの分散特徴を重み付き和で統合し、blurriness_raw（高いほど良い）を算出する。6) config の discretize_thresholds_raw（bad/poor/fair/good の4境界）で 5段階（0/0.25/0.5/0.75/1.0）に離散化し、blurriness_score と blurriness_grade を返す。

---

## 5. Normalization Policy

正規化の基本方針は「入力スケール差を評価前に吸収し、特徴量は 0..1 輝度空間で算出する」。具体的には (a) BGR/Gray + uint8/uint16/float を gray01（0..1）へ正規化、(b) Sobel/Laplacian/Diff を float32 で算出、(c) 分散（var）を raw として扱う。これにより 0..255 依存の桁ズレを避け、しきい値（discretize_thresholds_raw）は 1e-4〜1e-2 程度の桁で運用される。

---

## 6. Tuning Integration

分布チューニング（score_dist_tune）は、評価結果CSVの blurriness_raw を対象に、ok行（必要に応じて non-fallback）を母集団として quantile ベースの4境界（12.5/32.5/62.5/87.5%）を算出し、config（blurriness.discretize_thresholds_raw）へ反映する。今回の最新データ（150 samples, all ok）からは bad=0.000683, poor=0.001041, fair=0.002492, good=0.004098 を採用した。

---

## 7. Fallback Policy

入力が ndarray でない、もしくは size==0 の場合は invalid として安全なデフォルト出力を返す（blurriness_eval_status="invalid"、理由を blurriness_fallback_reason に記録）。画像が min_size 未満の場合は fallback として扱い（blurriness_eval_status="fallback"、reason="too_small_image_{w}x{h}"）、診断値（raw/variance）とスコアは算出しつつ、信頼度が低いことを status/reason で明示する。raw が非有限の場合は fallback に遷移し reason="non_finite_raw" を記録する。

---

## 8. Testing

ユニットテストで以下を保証する：1) 基本出力の存在（blurriness_score/blurriness_raw/blurriness_grade/blurriness_eval_status 等）。2) 離散スコアの妥当性（{0,0.25,0.5,0.75,1.0}に収まる）。3) 入力異常耐性（ndarray以外／size0 で必ず安全な出力を返し、status/reason が入る）。4) コントラクト出力（blurriness_raw_direction="higher_is_better"、blurriness_raw_transform="identity"、blurriness_higher_is_better=True）が常に欠損しない。5) しきい値読み込みが config をSSOTとし、単調性（bad<=poor<=fair<=good）が維持される（sortedで事故防止）。

---

## 9. Future Plans

1) face_blurriness（顔領域）で同等の raw/score/contract を整備し、全体との整合を保つ。2) 画像サイズが極端に小さいケースの扱い（fallbackでもscoreを出す/出さない）を運用方針として固定し、tune母集団フィルタ（ok & non-fallback）を標準化する。3) 特徴量のロバスト化（clip_percentile の運用調整、heavy outlier 検知）と、極端条件（夜景・逆光・高ISO）のプロファイル化を検討。4) 分布チューニングの対象期間・対象ファイル選定ルールを明文化し、旧スケール混在を防ぐ（rawスケール契約の維持）。

---

## 10. Design Philosophy

Raw（診断）とScore（運用）を分離し、スコアは比較・合成に強い離散値へ正規化する。異常入力・条件不足を黙殺せず、必ず eval_status と fallback_reason で理由を残す。「落ちない」「追える」「揺れにくい」を優先する。raw の向きは全指標で統一（高いほど良い）し、direction/transform/higher_is_better を常に欠損なく出力する。

---

## 11. Change History

| Date | Change | Author |
|------|--------|--------|
| 2026-02 | raw契約（higher_is_better）と contract keys（raw_direction/raw_transform/higher_is_better）を常時出力する形に統一。dtype/スケール差吸収のため gray01（0..1）正規化を固定し、Sobel/Laplacian/Diff の分散特徴を統合した raw を採用。 | akky0108 |
| 2026-02 | 分布チューニング（score_dist_tune）により discretize_thresholds_raw を再推定。最新データ（150 samples, all ok）で quantile (12.5/32.5/62.5/87.5) を採用し、bad=0.000683, poor=0.001041, fair=0.002492, good=0.004098 をSSOTとして反映。 | akky0108 |
