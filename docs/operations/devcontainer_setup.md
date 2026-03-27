# Dev Container セットアップ手順（photo_insight）

## 概要

本ドキュメントは、`photo_insight` の開発環境を Dev Container（Docker + VS Code）で構築・利用する手順をまとめたものです。

本構成により、以下を実現します。

* 開発環境の完全再現
* Python / 依存ライブラリの統一
* VS Code 拡張の自動適用
* Git / GitHub CLI の統合利用

---

## 前提条件

以下がインストール済みであること。

* Docker
* Docker Compose
* VS Code
* Dev Containers 拡張（VS Code）

---

## 初回セットアップ手順

### 1. リポジトリをクローン

```bash
git clone <repository-url>
cd photo_insight
```

---

### 2. Dev Container を起動

VS Code でプロジェクトを開き、以下を実行。

```
Ctrl + Shift + P
→ Dev Containers: Reopen in Container
```

または

```
Dev Containers: Rebuild and Reopen in Container
```

---

### 3. 初回ビルド確認

コンテナー起動後、以下を実行。

```bash
python -V
pip -V
ruff --version
pytest --version
gh --version
```

---

## GitHub CLI のセットアップ

初回のみ認証を実施。

```bash
gh auth login
```

確認：

```bash
gh auth status
```

---

## よく使うコマンド

### テスト実行

```bash
pytest -q tests
```

### Lint / Format 確認

```bash
ruff check .
ruff format .
```

### GitHub 操作

```bash
gh issue list
gh pr status
gh pr create
```

---

## Dev Container の再構築

設定変更後は再ビルドが必要。

```
Ctrl + Shift + P
→ Dev Containers: Rebuild Container
```

---

## ディレクトリ構成とマウント

| パス                        | 説明               |
| ------------------------- | ---------------- |
| `/work`                   | プロジェクトルート        |
| `/work/src`               | ソースコード           |
| `/home/vscode/.cache/pip` | pip キャッシュ        |
| `/home/vscode/.ssh`       | SSHキー（read-only） |
| `/home/vscode/.gitconfig` | Git設定（read-only） |

---

## 環境変数

主な設定値：

```bash
PYTHONPATH=/work/src
PROJECT_ROOT=/work
TZ=Asia/Tokyo
```

---

## トラブルシューティング

### コンテナーが起動しない

* Docker が起動しているか確認
* 再ビルドを実施

```
Dev Containers: Rebuild Container
```

---

### 拡張機能が反映されない

* devcontainer.json を確認
* 再ビルドを実施

---

### GitHub CLI が使えない

```bash
gh auth login
```

---

### SSH 接続エラー

* ホスト側の `~/.ssh` を確認
* パーミッション設定を確認

---

## 注意事項

* コンテナーは再ビルドでリセットされる
* 必要なツールは Dockerfile に追加すること
* 手動インストールは永続化されない

---

## 運用ルール（推奨）

* 開発は Dev Container を前提とする
* 環境差異は Dockerfile / devcontainer.json に反映する
* ローカル環境依存の設定を避ける

---

## 補足

本構成では以下を実現しています。

* Python 仮想環境：`/opt/venv`
* non-root ユーザー運用（vscode）
* pip キャッシュ永続化
* GitHub CLI 統合
* VS Code 拡張の自動適用

---
