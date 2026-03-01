#!/usr/bin/env bash
set -euo pipefail

PROD_ROOT=/home/mluser/photo_insight_prod
DST_DIR="$PROD_ROOT/config"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BASE="$REPO_ROOT/config/base/base.yaml"
ENV="$REPO_ROOT/config/env/prod.yaml"

BUILD_DIR="$REPO_ROOT/.build_config"
OUT_CFG="$BUILD_DIR/config.yaml"

need_file() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "ERROR: required file not found: $f" >&2
    exit 1
  fi
}

mkdir -p "$BUILD_DIR"
sudo mkdir -p "$DST_DIR"

# --- 必須チェック（ここが精緻化ポイント）---
need_file "$REPO_ROOT/scripts/render_config.py"
need_file "$BASE"
need_file "$ENV"

need_file "$REPO_ROOT/config/base/thresholds.yaml"
need_file "$REPO_ROOT/config/base/evaluator_thresholds.yaml"
need_file "$REPO_ROOT/config/base/logging_config.yaml"
need_file "$REPO_ROOT/config/base/quality_thresholds.yaml"
need_file "$REPO_ROOT/config/base/evaluation_rank.yaml"

# --- 生成 ---
python "$REPO_ROOT/scripts/render_config.py" "$BASE" "$ENV" "$OUT_CFG"
need_file "$OUT_CFG"

# --- バックアップ ---
ts=$(date +%Y%m%d_%H%M%S)
if [ -f "$DST_DIR/config.yaml" ]; then
  sudo cp -a "$DST_DIR/config.yaml" "$DST_DIR/config.yaml.bak.$ts"
fi

# --- 配布（composeが読むファイル名に合わせる）---
sudo install -m 0644 "$OUT_CFG" "$DST_DIR/config.yaml"

# --- baseリソース配布 ---
for f in thresholds.yaml evaluator_thresholds.yaml logging_config.yaml quality_thresholds.yaml evaluation_rank.yaml; do
  sudo install -m 0644 "$REPO_ROOT/config/base/$f" "$DST_DIR/$f"
done

sudo chmod -R a+rX "$DST_DIR"

echo "Deployed to: $DST_DIR"
ls -l "$DST_DIR" | sed -n '1,200p'