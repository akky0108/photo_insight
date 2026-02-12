# LocalContrastEvaluator Design & Contract

## 1. Overview

局所的な陰影の締まりやコントラスト不足（眠い写真）を検出する評価モジュール。

---

## 2. Output Contract

### 2.1 Output Fields

| Field | Type | Description |
|-------|------|-------------|
| local_contrast_raw | float | ブロック標準偏差の中央値（診断用。スケール依存の指標固有値） |
| local_contrast_score | float | 5段階離散スコア（比較・合成用の主指標。scale-invariant） |
| local_contrast_std | float | ブロックstdのばらつき（評価の安定性指標） |
| local_contrast_eval_status | str | 評価状態（ok / fallback_used / invalid_input） |
| local_contrast_fallback_reason | str | フォールバック発生時の理由ログ |
| success | bool | 評価処理の成否フラグ |


---

## 3. Purpose

露出や入力スケール（0..255 / 0..1）に依存せず、陰影の弱さ・立体感の欠如を安定して検出する。

---

## 4. Algorithm

Tile (block) the grayscale image and compute per-block standard deviation. Aggregate with median (local_contrast_raw) and dispersion (local_contrast_std).

---

## 5. Normalization Policy

Scale-invariant normalization using robust range: robust_range = p95(gray) - p05(gray). ratio = local_contrast_raw / robust_range, then map ratio to 0..1 via (raw_floor..raw_ceil) and gamma correction.

---

## 6. Tuning Integration



---

## 7. Fallback Policy

有効ブロック不足・robust_range 異常・nonfinite 発生時は global_std/robust_range を用いてフォールバックし、必ず理由を記録する。robust_range 自体が作れない場合は最終手段として global_std のみで簡易正規化する。

---

## 8. Testing

入力スケール不変性（uint8 vs float01）・フォールバック安全性・異常入力耐性をユニットテストで保証する。

---

## 9. Future Plans

撮影条件別プロファイル対応、被写体領域（顔/肌/背景）別の重み付け、バッチ分布を用いた自動正規化への拡張を予定。

---

## 10. Design Philosophy

Rawは診断用、Scoreは運用用として分離し、静かな失敗を許容しない。

---

## 11. Change History

| Date | Change | Author |
|------|--------|--------|
| 2026-02 | Scale-invariant normalization via robust_range (p95-p05) and safety fallbacks; unit tests updated | Akky |
