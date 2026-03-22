#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# deploy_config.sh
# 本番用 config を生成・配置する
# ============================================================

# ---- configurable (override via env) ----
PROD_ROOT="${PROD_ROOT:-/home/mluser/photo_insight_prod}"
DST_DIR="${DST_DIR:-$PROD_ROOT/config}"

# scripts/deploy 配下前提
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

BASE="${BASE:-$REPO_ROOT/config/base/base.yaml}"
ENV_YAML="${ENV_YAML:-$REPO_ROOT/config/env/prod.yaml}"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/.build_config}"
OUT_CFG="${OUT_CFG:-$BUILD_DIR/config.yaml}"

# sudo を使うか（0=使わない / 1=必要時だけ使う）
USE_SUDO="${USE_SUDO:-1}"

need_file() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "ERROR: required file not found: $f" >&2
    exit 1
  fi
}

run() {
  if [ "${USE_SUDO}" = "1" ] && [ "${1:-}" = "sudo" ]; then
    shift
    sudo "$@"
  else
    "$@"
  fi
}

echo "[INFO] REPO_ROOT: $REPO_ROOT"

# ---- checks ----
need_file "$REPO_ROOT/scripts/render_config.py"
need_file "$BASE"
need_file "$ENV_YAML"

for f in thresholds.yaml evaluator_thresholds.yaml logging_config.yaml quality_thresholds.yaml evaluation_rank.yaml; do
  need_file "$REPO_ROOT/config/base/$f"
done

mkdir -p "$BUILD_DIR"

# DST_DIR はまず sudo 無しで作る。ダメなら sudo にフォールバック。
if mkdir -p "$DST_DIR" 2>/dev/null; then
  :
else
  if [ "${USE_SUDO}" = "1" ]; then
    sudo mkdir -p "$DST_DIR"
  else
    echo "ERROR: cannot create DST_DIR without sudo: $DST_DIR" >&2
    exit 1
  fi
fi

# ---- generate ----
echo "[INFO] generating config.yaml..."
python "$REPO_ROOT/scripts/render_config.py" "$BASE" "$ENV_YAML" "$OUT_CFG"
need_file "$OUT_CFG"

# ---- backup (best-effort) ----
ts=$(date +%Y%m%d_%H%M%S)
if [ -f "$DST_DIR/config.yaml" ]; then
  echo "[INFO] backup existing config.yaml"
  if cp -a "$DST_DIR/config.yaml" "$DST_DIR/config.yaml.bak.$ts" 2>/dev/null; then
    :
  else
    if [ "${USE_SUDO}" = "1" ]; then
      sudo cp -a "$DST_DIR/config.yaml" "$DST_DIR/config.yaml.bak.$ts"
    else
      echo "WARN: cannot backup without sudo: $DST_DIR/config.yaml" >&2
    fi
  fi
fi

# ---- deploy (prefer non-sudo; fallback if needed) ----
echo "[INFO] deploying config.yaml"
install -m 0644 "$OUT_CFG" "$DST_DIR/config.yaml" 2>/dev/null \
  || ( [ "${USE_SUDO}" = "1" ] && sudo install -m 0644 "$OUT_CFG" "$DST_DIR/config.yaml" )

echo "[INFO] deploying base config files"
for f in thresholds.yaml evaluator_thresholds.yaml logging_config.yaml quality_thresholds.yaml evaluation_rank.yaml; do
  install -m 0644 "$REPO_ROOT/config/base/$f" "$DST_DIR/$f" 2>/dev/null \
    || ( [ "${USE_SUDO}" = "1" ] && sudo install -m 0644 "$REPO_ROOT/config/base/$f" "$DST_DIR/$f" )
done

# 読めれば OK なので、chmod は基本不要。必要なら最小限に。
chmod -R a+rX "$DST_DIR" 2>/dev/null \
  || ( [ "${USE_SUDO}" = "1" ] && sudo chmod -R a+rX "$DST_DIR" )

echo "[OK] deployed to: $DST_DIR"
ls -la "$DST_DIR" | sed -n '1,200p'