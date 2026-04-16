# photo_insight 本番デプロイ手順書

## 1. 目的

本ドキュメントは `photo_insight` の本番環境（WSL2 + Docker）における
**デプロイおよび実行手順を標準化すること**を目的とする。

対象:

* 設定ファイルの生成・配置
* Docker イメージの更新
* 本番処理の実行
* トラブル時の確認手順

---

## 2. 対象環境

* OS: WSL2 (Ubuntu)
* 実行基盤: Docker / Docker Compose
* 本番ディレクトリ: `~/photo_insight_prod`
* リポジトリ: `~/photo_insight`

---

## 3. ディレクトリ構成（本番）

```
photo_insight_prod/
├── compose.prod.yaml
├── config/
├── runs/
├── logs/
├── tmp/
├── output/
```

---

## 4. スクリプト構成（推奨）

※配置変更OK前提で整理

```
scripts/
├── deploy/
│   ├── deploy_prod.sh
│   ├── deploy_prod_config.sh
│   └── release_prod_image.sh
├── ops/
│   └── run_prod.sh
├── debug/
│   └── (調査用スクリプト)
└── render_config.py
```

---

## 5. スクリプト役割

### 5.1 render_config.py

* base.yaml + prod.yaml をマージ
* config.yaml を生成

### 5.2 deploy_prod_config.sh

* 本番用 config 一式を生成・配置
* 以下をコピー対象とする:

  * config.yaml
  * thresholds.yaml
  * evaluator_thresholds.yaml
  * logging_config.yaml

### 5.3 deploy_prod.sh

* 本番 Docker image の build / 更新
* tag 管理

### 5.4 release_prod_image.sh

* image の release 用補助スクリプト
* tag 固定やバージョン管理に使用

### 5.5 run_prod.sh

* 本番実行のエントリポイント
* docker compose run をラップ

### 5.6 debug/*

* 障害調査専用
* 通常運用では使用しない

---

## 6. 標準デプロイ手順

### 6.1 リポジトリ更新

```
cd ~/photo_insight
git pull origin develop
```

---

### 6.2 設定ファイル生成・配置

```
bash scripts/deploy/deploy_prod_config.sh
```

確認:

```
ls ~/photo_insight_prod/config/
```

---

### 6.3 本番イメージ更新

```
bash scripts/deploy/deploy_prod.sh
```

または（必要に応じて）

```
bash scripts/deploy/release_prod_image.sh
```

---

### 6.4 本番実行

```
bash scripts/ops/run_prod.sh
```

---

## 7. 実行確認

### 7.1 コンテナ起動確認

```
docker ps -a
```

---

### 7.2 ログ確認

```
tail -n 100 ~/photo_insight_prod/logs/*.log
```

---

### 7.3 出力確認

```
ls ~/photo_insight_prod/output
ls ~/photo_insight_prod/runs
```

---

## 8. 運用ルール

### 8.1 build と run の分離

* 本番では build しない
* run は既存 image を使用する

---

### 8.2 config は手編集しない

禁止:

```
config.yaml を直接編集
```

必ず:

```
render_config.py 経由で生成
```

---

### 8.3 実行は run_prod.sh を使用

禁止:

```
docker compose を手打ち
```

理由:

* ミス防止
* 引数統一

---

## 9. ロールバック手順

### 9.1 image 戻し

```
export PHOTO_INSIGHT_IMAGE_TAG=previous
```

---

### 9.2 config 戻し

```
cp backup/config.yaml ~/photo_insight_prod/config/
```

---

### 9.3 再実行

```
bash scripts/ops/run_prod.sh
```

---

## 10. トラブルシュート

### 10.1 logging_config.yaml が見つからない

エラー:

```
Config file not found
```

対応:

```
/work/config/logging_config.yaml が存在するか確認
```

---

### 10.2 input mount 不一致

確認:

```
docker compose config
```

---

### 10.3 config 未反映

確認:

```
cat ~/photo_insight_prod/config/config.yaml
```

---

### 10.4 image tag 不一致

確認:

```
docker images | grep photo_insight
```

---

### 10.5 実行パスミス

NG例:

```
compose.prod.yaml が存在しないパスで実行
```

---

## 11. 今後の改善ポイント

* pipeline chain（nef → portrait_quality）統合
* 日付指定（--date）の標準化
* incremental 処理の共通化
* GPU対応

---

## 12. 標準運用フロー（まとめ）

1. リポジトリ更新
2. config 生成（deploy_prod_config.sh）
3. image 更新（deploy_prod.sh）
4. 実行（run_prod.sh）
5. ログ・出力確認

---

以上
    