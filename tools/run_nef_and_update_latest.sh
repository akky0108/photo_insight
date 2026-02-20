#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-}"
TARGET_DIR="${2:-}"
CONFIG="${3:-config/config.prod.yaml}"
MAX_WORKERS="${4:-2}"

usage() {
  echo "Usage: $0 <session:YYYY-MM-DD> <target_dir> [config] [max_workers]"
}

# ----------------------------
# validate args
# ----------------------------
if [[ -z "${SESSION}" || -z "${TARGET_DIR}" ]]; then
  usage
  exit 2
fi

if [[ ! "${SESSION}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "ERROR: session must be YYYY-MM-DD: '${SESSION}'"
  usage
  exit 2
fi

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "ERROR: target_dir does not exist or is not a directory: '${TARGET_DIR}'"
  exit 2
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: '${CONFIG}'"
  exit 2
fi

# ----------------------------
# run
# ----------------------------
echo "[info] session=${SESSION}"
echo "[info] target_dir=${TARGET_DIR}"
echo "[info] config=${CONFIG}"
echo "[info] max_workers=${MAX_WORKERS}"

python -m photo_insight.cli.run_batch \
  --processor nef \
  --config "${CONFIG}" \
  --max-workers "${MAX_WORKERS}" \
  --target-dir "${TARGET_DIR}"

# ----------------------------
# update latest (with verify)
# ----------------------------
tools/update_latest_nef.sh "${SESSION}" --verify

# ----------------------------
# quick sanity check output
# ----------------------------
LATEST_DIR="runs/latest/nef/${SESSION}"
LATEST_CSV="${LATEST_DIR}/${SESSION}_raw_exif_data.csv"

echo "[ok] done. latest dir:"
ls -la "${LATEST_DIR}" || true

if [[ -f "${LATEST_CSV}" ]]; then
  echo "[ok] latest csv:"
  echo "  path: ${LATEST_CSV}"
  echo -n "  lines: "
  wc -l "${LATEST_CSV}" | awk '{print $1}'
  echo -n "  header: "
  head -n 1 "${LATEST_CSV}"
else
  echo "WARN: latest csv not found: ${LATEST_CSV}"
fi