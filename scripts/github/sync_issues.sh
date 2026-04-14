#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DEFAULT_ISSUES_YML="$REPO_ROOT/.github/issues/epics.yaml"
LEGACY_ISSUES_YML="$REPO_ROOT/docs/operations/github/issues.yml"

if [[ -n "${ISSUES_YML:-}" ]]; then
  RESOLVED_ISSUES_YML="$ISSUES_YML"
elif [[ -f "$DEFAULT_ISSUES_YML" ]]; then
  RESOLVED_ISSUES_YML="$DEFAULT_ISSUES_YML"
elif [[ -f "$LEGACY_ISSUES_YML" ]]; then
  RESOLVED_ISSUES_YML="$LEGACY_ISSUES_YML"
else
  RESOLVED_ISSUES_YML="$DEFAULT_ISSUES_YML"
fi

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
TARGET_REPO="${TARGET_REPO:-akky0108/photo_insight}"

echo "[INFO] repo: $TARGET_REPO"
echo "[INFO] issues file: $RESOLVED_ISSUES_YML"
if [[ -f "$ENV_FILE" ]]; then
  echo "[INFO] env file: $ENV_FILE"
fi

ARGS=(
  "$REPO_ROOT/scripts/github/sync_issues.py"
  --repo "$TARGET_REPO"
  --issues-yml "$RESOLVED_ISSUES_YML"
)

if [[ -f "$ENV_FILE" ]]; then
  ARGS+=(--env-file "$ENV_FILE")
fi

for arg in "$@"; do
  ARGS+=("$arg")
done

exec "$PYTHON_BIN" "${ARGS[@]}"