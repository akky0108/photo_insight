# native crash 再現条件固定

## 対象
[2026-0011] native crash 再現条件固定

## 検証環境
- devcontainer
- CPUExecutionProvider
- config: /work/config/config.repro.yaml

## 検証入力
- date: 2026-02-17
- input dir: /work/input/2026/2026-02-17
- tested files: 4 NEF

## 実施コマンド
- max_workers=1, max_images=1
- max_workers=1, max_images=4
- max_workers=2, max_images=4
- max_workers=4, max_images=4

## 結果
- いずれも native crash は未再現
- pipeline は success
- rawpy / evaluator init / evaluator.evaluate は通過
- 最大観測メモリ使用量: 約 3.15 GB

## 補足
- max_images の厳密適用に不整合が見られるケースあり
- 再現が必要な場合は本番環境差分・大量件数・特定ファイル依存を優先確認