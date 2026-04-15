#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${BASE_BRANCH:-main}"
SOURCE_BRANCH="${SOURCE_BRANCH:-develop}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
DRAFT_MODE="${DRAFT_MODE:-false}"

usage() {
  cat <<'EOF'
develop → main のリリース PR を作成する。

Usage:
  ./scripts/github/create_release_pr.sh [--draft]

Options:
  --draft   Draft PR を作成する

Environment:
  BASE_BRANCH=main
  SOURCE_BRANCH=develop
  REMOTE_NAME=origin
EOF
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $cmd" >&2
    exit 1
  fi
}

ensure_auth() {
  if ! gh auth status >/dev/null 2>&1; then
    echo "[ERROR] gh is not authenticated. Run: gh auth login" >&2
    exit 1
  fi
}

has_diff() {
  local diff_count
  diff_count="$(git rev-list --count "${REMOTE_NAME}/${BASE_BRANCH}..${REMOTE_NAME}/${SOURCE_BRANCH}")"
  [[ "$diff_count" -gt 0 ]]
}

ensure_no_existing_pr() {
  local existing
  existing="$(gh pr list \
    --base "$BASE_BRANCH" \
    --head "$SOURCE_BRANCH" \
    --json number \
    -q '.[0].number' 2>/dev/null || true)"

  if [[ -n "$existing" ]]; then
    echo "[ERROR] PR already exists: #$existing"
    exit 1
  fi
}

build_pr_title() {
  local now
  now="$(date '+%Y-%m-%d')"
  echo "Release: develop → main ($now)"
}

build_pr_body() {
  cat <<EOF
## 概要
develop の変更を main に反映するリリース PR

## 内容
- develop → main の差分を統合

## 確認
- [ ] CI が通過している
- [ ] 変更内容を確認済み

## 備考
- 自動生成 PR
EOF
}

main() {
  local draft_flag="$DRAFT_MODE"

  if [[ "${1:-}" == "--draft" ]]; then
    draft_flag="true"
  fi

  require_command git
  require_command gh
  ensure_auth

  echo "[INFO] Fetching latest..."
  git fetch "$REMOTE_NAME" --prune

  echo "[INFO] Checking diff: $SOURCE_BRANCH → $BASE_BRANCH"
  if ! has_diff; then
    echo "[INFO] No diff. Release PR not needed."
    exit 0
  fi

  ensure_no_existing_pr

  local title
  title="$(build_pr_title)"

  local body
  body="$(build_pr_body)"

  echo "[INFO] Creating release PR..."

  if [[ "$draft_flag" == "true" ]]; then
    gh pr create \
      --base "$BASE_BRANCH" \
      --head "$SOURCE_BRANCH" \
      --title "$title" \
      --body "$body" \
      --draft
  else
    gh pr create \
      --base "$BASE_BRANCH" \
      --head "$SOURCE_BRANCH" \
      --title "$title" \
      --body "$body"
  fi

  echo "[OK] Release PR created."
}

main "$@"