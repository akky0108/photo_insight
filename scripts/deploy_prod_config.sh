#!/usr/bin/env bash
set -euo pipefail

PROD_ROOT=/home/mluser/photo_insight_prod
DST_DIR="$PROD_ROOT/config"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BASE="$REPO_ROOT/config/base/base.yaml"
ENV="$REPO_ROOT/config/env/prod.yaml"

# 生成先（一時）
BUILD_DIR="$REPO_ROOT/.build_config"
OUT_CFG="$BUILD_DIR/config.yaml"

mkdir -p "$BUILD_DIR"

# base + env をマージして config.yaml を生成
python "$REPO_ROOT/scripts/render_config.py" "$BASE" "$ENV" "$OUT_CFG"

sudo mkdir -p "$DST_DIR"

ts=$(date +%Y%m%d_%H%M%S)
if [ -f "$DST_DIR/config.yaml" ]; then
  sudo cp -a "$DST_DIR/config.yaml" "$DST_DIR/config.yaml.bak.$ts"
fi

# 生成した config.yaml を配置（composeが読むファイル名）
sudo install -m 0644 "$OUT_CFG" "$DST_DIR/config.yaml"

# 共有リソースも配置（必要なものだけでOK）
sudo install -m 0644 "$REPO_ROOT/config/base/thresholds.yaml" "$DST_DIR/thresholds.yaml"

for f in logging_config.yaml evaluator_thresholds.yaml quality_thresholds.yaml evaluation_rank.yaml; do
  if [ -f "$REPO_ROOT/config/base/$f" ]; then
    sudo install -m 0644 "$REPO_ROOT/config/base/$f" "$DST_DIR/$f"
  fi
done

sudo chmod -R a+rX "$DST_DIR"

echo "Deployed to: $DST_DIR"
ls -l "$DST_DIR" | sed -n '1,200p'
