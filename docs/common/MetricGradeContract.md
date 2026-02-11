# Metric Grade Contract

## Overview
全 evaluator/mapper/acceptance/分析ツールに跨る grade の SSOT を定義する。

## Grade enum (SSOT)
- `bad`
- `poor`
- `fair`
- `good`
- `excellent`


## Score to grade mapping
| score | grade |
|---:|---|
| 0.0 | `bad` |
| 0.25 | `poor` |
| 0.5 | `fair` |
| 0.75 | `good` |
| 1.0 | `excellent` |


## eval_status policy
- `ok`: 通常評価。grade は score と整合する。
- `fallback`: 評価は継続するが信頼度が低い。grade は score と整合し、fallback_reason を必須。
- `invalid`: 入力不正など。score/grade は既定値を許容し、fallback_reason を必須。


## Legacy grade aliases (normalization)
- `very_blurry` -> `bad`
- `blurry` -> `poor`


## Notes
- face_ の付与は mapper 側の責務
- CSV には score/grade/eval_status/fallback_reason を出す
