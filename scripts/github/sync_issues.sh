#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ISSUES_YML="${ISSUES_YML:-$REPO_ROOT/docs/operations/github/issues.yml}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
TARGET_REPO="${TARGET_REPO:-akky0108/photo_insight}"

ARGS=(
  "$REPO_ROOT/scripts/github/sync_issues.py"
  --repo "$TARGET_REPO"
  --issues-yml "$ISSUES_YML"
)

if [[ -f "$ENV_FILE" ]]; then
  ARGS+=(--env-file "$ENV_FILE")
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  ARGS+=(--dry-run)
fi

exec "$PYTHON_BIN" "${ARGS[@]}"