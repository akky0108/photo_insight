# パイプライン仕様書：NEF → portrait_quality 2段 Chain

## 1. 目的

本ドキュメントは `photo_insight` における **NEF → portrait_quality の2段パイプライン実行仕様**を定義する。

目的は以下の通り。

* CLI から **1コマンドで2段処理を実行**できるようにする
* 前段成功時のみ後段を実行する
* NEF の出力 CSV を portrait_quality の正式入力として定義する
* `date / target_dir / incremental / max_images` をパイプライン文脈として統一する
* `BaseBatchProcessor` の責務を肥大化させない

---

# 2. スコープ

## 含む

* CLI `--pipeline` 実行
* `nef → portrait_quality` 2段 chain
* stage 間 artifact 契約
* pipeline context 伝播
* 失敗時停止ルール
* pipeline summary 出力

## 含まない

* 3段以上の pipeline
* 任意順序 pipeline
* stage 個別 max-images
* BaseBatchProcessor 再設計

---

# 3. 用語

## Stage

単独で実行可能な処理単位。

本仕様では

* `nef`
* `portrait_quality`

を指す。

## Pipeline

複数 stage を順序付きで実行する CLI 機能。

## Artifact

stage の出力で後段が利用する成果物。

例

* CSV
* run directory
* processed_images

## Pipeline Context

pipeline 全体で共有される実行パラメータ。

```
date
target_dir
incremental
max_images
config_path
```

## Stage Result

stage 実行後に orchestration 層へ返されるメタ情報。

---

# 4. 設計方針

pipeline 制御は **CLI / orchestration 層が担当**する。

processor は **単段処理責務のみ持つ。**

## CLI の責務

* pipeline 解析
* stage 実行順序制御
* artifact 受け渡し
* failure control
* pipeline summary

## Processor の責務

* 計算処理
* 入力 artifact 解釈
* 出力 artifact 生成

`BaseBatchProcessor` は **pipeline を管理しない。**

---

# 5. CLI 仕様

## 実行例

```
python -m photo_insight.cli.run_batch \
  --pipeline nef,portrait_quality \
  --date 2026-02-17
```

## 新規オプション

`--pipeline`

カンマ区切りで stage を指定する。

例

```
nef,portrait_quality
```

## 既存オプションとの関係

`--pipeline` と `--processor` は **排他**。

ルール

* 両方指定 → エラー
* 両方未指定 → エラー

## 対応 pipeline

現時点では以下のみ。

```
nef,portrait_quality
```

---

# 6. Pipeline 実行フロー

```
nef
 ↓
portrait_quality
```

処理手順

1. `nef` 実行
2. 成功 → `portrait_quality` 実行
3. 失敗 → pipeline 停止

pipeline success 条件

**全 stage 成功**

---

# 7. Artifact 契約

`nef` の出力 CSV を

**portrait_quality の正式入力**とする。

運用参照位置

```
runs/latest/nef/<session>/evaluation_results.csv
```

ただし pipeline 実行中は

**latest 再探索は行わない。**

代わりに

```
stage_result.output_csv_path
```

を直接後段へ渡す。

## 必須 artifact

* name
* status
* session_id
* run_dir
* output_csv_path
* date
* target_dir

## 任意

* processed_images_path
* processed_count
* skipped_count
* error_count

---

# 8. Pipeline Context

CLI は以下を全 stage に伝播する。

```
--date
--target-dir
--incremental
--max-images
--config
```

stage は解釈するが

**context 制御は CLI が担当する。**

---

# 9. max-images

`--max-images` は

**pipeline 全体の処理上限**。

目的

* メモリ消費制御
* 実行時間制御
* 運用安全性

## ルール

```
nef 出力 ≤ max-images
portrait_quality 入力 ≤ nef 出力
```

## 例

### ケース1

```
max-images = 100
nef 出力 = 100
portrait_quality = 100
```

### ケース2

```
max-images = 100
実画像 = 32
nef 出力 = 32
portrait_quality = 32
```

### ケース3

```
resume
入力100
既処理60
実行40
```

stage 個別上限は **未対応**。

---

# 10. Pipeline Summary

pipeline 実行結果を JSON で出力する。

最低限含む項目

* pipeline
* date
* status
* stage status
* session_id
* input/output CSV

例

```
{
  "pipeline": ["nef", "portrait_quality"],
  "date": "2026-02-17",
  "status": "success",
  "stages": [
    {
      "name": "nef",
      "status": "success",
      "session_id": "...",
      "output_csv_path": "..."
    },
    {
      "name": "portrait_quality",
      "status": "success",
      "session_id": "...",
      "input_csv_path": "...",
      "output_csv_path": "..."
    }
  ]
}
```

stage 個別 summary は残してよい。

---

# 11. エラーハンドリング

ルール

```
前段失敗 → 後段実行しない
```

```
ne f failure
↓
portrait_quality skipped
```

pipeline success 条件

**全 stage success**

ログには以下を出す

* 失敗 stage
* skip stage
* エラー理由

---

# 12. 互換性

単段実行は維持する。

```
--processor nef
--processor portrait_quality
```

processor は

* 単独
* pipeline

両方で動作する必要がある。

portrait_quality は

```
input_csv_path
```

override を受け取れる必要がある。

---

# 13. 実装責務

## CLI

担当

* pipeline 解析
* stage 実行順序
* artifact 受け渡し
* summary

主実装

```
photo_insight/cli/run_batch.py
```

## Processor

担当

* 処理ロジック
* artifact 入出力

processor は

**他 processor を参照しない。**

## BaseBatchProcessor

以下は持たない

* pipeline
* stage dependency
* artifact routing

---

# 14. 変更対象ファイル

```
src/photo_insight/cli/run_batch.py
src/photo_insight/cli/processor_registry.py
src/photo_insight/pipelines/nef/
src/photo_insight/pipelines/portrait_quality/
```

BaseBatchProcessor は原則変更しない。

---

# 15. 推奨データ構造

```
class ChainContext:
    date: str | None
    target_dir: str | None
    incremental: bool | None
    max_images: int | None
    config_path: str | None
```

```
class StageResult:
    name: str
    status: str
    session_id: str | None
    run_dir: str | None
    input_csv_path: str | None
    output_csv_path: str | None
    processed_count: int | None
    skipped_count: int | None
    error_count: int | None
    error_message: str | None
```

---

# 16. 受け入れ条件

以下を満たすこと

1. `--pipeline nef,portrait_quality` が動作
2. `--processor` と排他
3. nef success → portrait_quality 実行
4. nef failure → portrait_quality skip
5. CSV 引き渡し
6. context 伝播
7. max-images 制御
8. pipeline summary 出力
9. 単段実行が壊れない

---

# 17. テスト方針

優先テスト

* CLI 引数排他
* pipeline parse
* success → 次 stage
* failure → stop
* artifact passing
* context 伝播
* summary

重いファイルテストより

**mock テストを優先。**

---

# 18. 将来拡張

候補

* pipeline 共通 context
* processed_images 標準化
* stage 個別 max-images
* multi stage pipeline
* framework artifact abstraction

---

# 19. 最終方針

Priority1 実装

* CLI で pipeline orchestration
* BaseBatchProcessor 変更最小
* NEF CSV を portrait_quality 正式入力
* max-images を workload guard とする
