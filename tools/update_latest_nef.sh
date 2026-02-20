#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Update runs/latest/nef/<session>/... from newest runs/*/.../artifacts/nef/<session>/
#
# Usage:
#   tools/update_latest_nef.sh 2026-02-17
#   tools/update_latest_nef.sh 2026-02-17 --dry-run
#   tools/update_latest_nef.sh 2026-02-17 --verify
#   tools/update_latest_nef.sh 2026-02-17 --dry-run --verify
#
# Notes:
# - Finds newest CSV under: runs/**/artifacts/nef/<session>/*_raw_exif_data.csv
# - Copies into: runs/latest/nef/<session>/
# - Uses atomic update via tmp dir + move files
# ----------------------------------------

SESSION="${1:-}"
shift || true

if [[ -z "${SESSION}" ]]; then
  echo "ERROR: session is required."
  echo "Usage: $0 <session_name> [--dry-run] [--verify]"
  exit 2
fi

DRY_RUN=0
VERIFY=0

# parse options (order-insensitive)
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --dry-run) DRY_RUN=1 ;;
    --verify)  VERIFY=1 ;;
    -h|--help)
      echo "Usage: $0 <session_name> [--dry-run] [--verify]"
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: ${1}"
      echo "Usage: $0 <session_name> [--dry-run] [--verify]"
      exit 2
      ;;
  esac
  shift
done

# Detect repo root (assumes this script is under tools/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEARCH_GLOB="runs"
DEST_DIR="runs/latest/nef/${SESSION}"

# Find newest matching CSV file for the session
# Example path:
# runs/2026-02-19/neffilebatchprocess_20260219_235117_59515/artifacts/nef/2026-02-17/2026-02-17_raw_exif_data.csv
NEWEST_LINE="$(
  find "${SEARCH_GLOB}" -type f \
    -path "*/artifacts/nef/${SESSION}/*" \
    -name "*_raw_exif_data.csv" \
    -printf "%T@ %p\n" \
  | sort -nr \
  | head -n 1 \
)"

if [[ -z "${NEWEST_LINE}" ]]; then
  echo "ERROR: No NEF CSV found for session='${SESSION}'."
  echo "Searched: ${SEARCH_GLOB}/**/artifacts/nef/${SESSION}/*_raw_exif_data.csv"
  exit 3
fi

NEWEST_PATH="${NEWEST_LINE#* }"
NEWEST_TS="${NEWEST_LINE%% *}"
SRC_DIR="$(dirname "${NEWEST_PATH}")"

echo "[info] repo_root: ${REPO_ROOT}"
echo "[info] session: ${SESSION}"
echo "[info] newest_csv: ${NEWEST_PATH}"
echo "[info] newest_mtime_epoch: ${NEWEST_TS}"
echo "[info] dest_dir: ${DEST_DIR}"

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "[dry-run] Would update latest from: ${NEWEST_PATH}"
  if [[ ${VERIFY} -eq 1 ]]; then
    echo "[dry-run] Would also verify latest matches newest after copy."
  fi
  exit 0
fi

# Prepare destination (atomic update)
TMP_DIR="${DEST_DIR}.tmp.$$"
mkdir -p "${TMP_DIR}"
mkdir -p "${DEST_DIR}"

cleanup() {
  rm -rf "${TMP_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[info] copying from src_dir: ${SRC_DIR}"

# Avoid failure when glob matches nothing
shopt -s nullglob
csvs=( "${SRC_DIR}/"*.csv )
shopt -u nullglob

if [[ ${#csvs[@]} -eq 0 ]]; then
  echo "ERROR: No CSV files found in src_dir: ${SRC_DIR}"
  exit 4
fi

cp -a "${csvs[@]}" "${TMP_DIR}/"

# Replace contents (CSV only)
rm -f "${DEST_DIR}/"*.csv 2>/dev/null || true
mv "${TMP_DIR}/"*.csv "${DEST_DIR}/"

echo "[ok] updated: ${DEST_DIR}"
echo "[ok] files:"
ls -la "${DEST_DIR}"

# -------------------------
# Verify (optional)
# -------------------------
if [[ ${VERIFY} -eq 1 ]]; then
  LATEST_CSV="${DEST_DIR}/${SESSION}_raw_exif_data.csv"
  echo "[verify] latest_csv: ${LATEST_CSV}"

  # 1) existence
  if [[ ! -f "${LATEST_CSV}" ]]; then
    echo "[ng] latest csv not found: ${LATEST_CSV}" >&2
    exit 10
  fi

  # 2) non-empty rows (>=2: header + 1 data row)
  lines="$(wc -l < "${LATEST_CSV}" | tr -d ' ')"
  if [[ "${lines}" -lt 2 ]]; then
    echo "[ng] latest csv is too small (lines=${lines}): ${LATEST_CSV}" >&2
    exit 11
  fi
  echo "[ok] lines=${lines}"

  # 3) newest vs latest strict match
  if ! diff -q "${LATEST_CSV}" "${NEWEST_PATH}" >/dev/null; then
    echo "[ng] latest != newest" >&2
    echo "      latest: ${LATEST_CSV}" >&2
    echo "      newest: ${NEWEST_PATH}" >&2
    exit 12
  fi
  echo "[ok] latest matches newest"
fi
