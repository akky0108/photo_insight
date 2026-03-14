# photo_insight Pipeline Architecture

## 概要

本ドキュメントは **photo_insight の Pipeline
アーキテクチャ**を説明する。

photo_insight では、複数のバッチ処理を **CLI
によるパイプラインオーケストレーション**で接続する設計を採用している。

この設計の目的は以下である。

-   処理の分離
-   パイプラインの拡張性確保
-   BaseBatchProcessor の責務最小化
-   Artifact 契約によるステージ接続
-   CLI による実行制御

------------------------------------------------------------------------

# 1. 全体アーキテクチャ

システムは以下のレイヤーで構成される。

CLI Layer\
Pipeline Orchestration Layer\
Processor Layer\
Artifact Layer

実行イメージ

CLI ↓ run_batch.py ↓ Pipeline Chain ↓ Processor A → Processor B ↓
Artifacts

------------------------------------------------------------------------

# 2. CLI Layer

CLI は pipeline の実行制御を担当する。

主な機能

-   pipeline 解析
-   processor 解決
-   runtime override 抽出
-   stage 実行順序制御
-   artifact 受け渡し
-   pipeline summary 出力

主な CLI

python -m photo_insight.cli.run_batch

例

python -m photo_insight.cli.run_batch\
--pipeline nef,portrait_quality\
--date 2026-02-17

------------------------------------------------------------------------

# 3. Pipeline Orchestration

pipeline orchestration は CLI 側で実装される。

責務

-   stage の順序制御
-   stage result の収集
-   artifact routing
-   error stop
-   pipeline summary 作成

処理フロー

Stage A ↓ Stage B

ルール

-   前段成功時のみ後段を実行
-   前段失敗時は pipeline 停止

------------------------------------------------------------------------

# 4. Processor Layer

Processor は **単段処理責務のみ持つ**。

Processor は以下のみ担当する。

-   データ処理
-   入力 artifact 読み込み
-   出力 artifact 生成

Processor は **他の processor を知らない。**

------------------------------------------------------------------------

# 5. Artifact Contract

stage 間の接続は **artifact 契約**で行う。

例

NEF stage

output_csv_path

↓

portrait_quality stage

input_csv_path

この方式により

-   stage の独立性
-   pipeline の拡張性

が確保される。

------------------------------------------------------------------------

# 6. Runtime Context

CLI から pipeline へ runtime context が渡される。

例

date target_dir max_images config_path

runtime context は CLI が管理し、processor に注入する。

------------------------------------------------------------------------

# 7. max_images 制御

max_images は **pipeline workload guard**として扱う。

ルール

-   pipeline 先頭 stage にのみ適用
-   後続 stage は upstream artifact に従う

例

max_images = 100

Stage A output = 100 Stage B input = 100

------------------------------------------------------------------------

# 8. Pipeline Summary

pipeline 実行結果は summary として出力される。

summary 内容

-   pipeline
-   status
-   stage results
-   artifact paths
-   processed count

summary は JSON artifact として保存される。

例

pipeline_summary.json

------------------------------------------------------------------------

# 9. エラーハンドリング

基本ルール

前段 failure → 後段 skip

例

Stage A failure ↓ Stage B skipped

pipeline success 条件

全 stage success

------------------------------------------------------------------------

# 10. 単段実行

pipeline 以外に **単段 processor 実行**もサポートする。

例

--processor nef

この設計により

-   processor 単体テスト
-   pipeline 実行

両方が可能になる。

------------------------------------------------------------------------

# 11. 設計の重要ポイント

今回の設計では以下を重要視している。

1.  BaseBatchProcessor を変更しない
2.  pipeline logic を CLI に集約
3.  processor の独立性
4.  artifact contract による接続

この構造により pipeline 拡張が容易になる。

------------------------------------------------------------------------

# 12. 将来拡張

将来的には以下の拡張が可能である。

-   multi-stage pipeline
-   pipeline resume
-   pipeline history
-   artifact abstraction
-   distributed pipeline

------------------------------------------------------------------------

# 13. まとめ

photo_insight の pipeline は

CLI orchestrated pipeline

という構造を採用している。

この設計により

-   processor の独立性
-   pipeline の拡張性
-   framework の安定性

が確保されている。
