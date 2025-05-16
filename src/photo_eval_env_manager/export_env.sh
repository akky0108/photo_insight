#!/bin/bash

# photo_eval_env をアクティブにしてから実行すること
# conda activate photo_eval_env

set -e  # エラーで即終了
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

BASE_YML="$SCRIPT_DIR/environment_base.yml"
PIP_TXT="$SCRIPT_DIR/pip_requirements.txt"
FINAL_YML="$PROJECT_DIR/environment_combined.yml"

# Conda初期化チェック
if ! command -v conda &> /dev/null; then
  echo "⚠️ Condaがインストールされていません。インストールしてください。"
  exit 1
fi

# 仮想環境がアクティブかチェック
if [ "$CONDA_DEFAULT_ENV" != "photo_eval_env" ]; then
  echo "⚠️ photo_eval_env 仮想環境がアクティブではありません。まず 'conda activate photo_eval_env' を実行してください。"
  exit 1
fi

# Conda環境のエクスポート
echo "📦 Conda環境をエクスポート中..."
conda env export --from-history > "$SCRIPT_DIR/exported_from_conda.yml"

# pip freeze
echo "📦 pipパッケージを freeze 中..."
pip freeze > "$PIP_TXT"

# merge_envs.pyが存在するか確認
if [ ! -f "$SCRIPT_DIR/merge_envs.py" ]; then
  echo "❌ merge_envs.py が見つかりません。"
  exit 1
fi

# YAMLのマージ
echo "🔧 YAML をマージ中..."
python3 "$SCRIPT_DIR/merge_envs.py" "$BASE_YML" "$PIP_TXT" "$FINAL_YML"

echo "✅ 完了！ -> $FINAL_YML が生成されました。"
