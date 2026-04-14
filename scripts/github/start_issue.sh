#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${BASE_BRANCH:-develop}"
DEFAULT_BRANCH_TYPE="${DEFAULT_BRANCH_TYPE:-fix}"
REMOTE_NAME="${REMOTE_NAME:-origin}"

usage() {
  cat <<'EOF'
Issue から作業ブランチを作成する。

Usage:
  ./scripts/github/start_issue.sh <issue_number> [branch_type]

Arguments:
  issue_number   GitHub Issue 番号
  branch_type    fix | feat | chore | refactor | docs | test
                 省略時は fix

Examples:
  ./scripts/github/start_issue.sh 123
  ./scripts/github/start_issue.sh 123 feat

Environment variables:
  BASE_BRANCH=develop
  DEFAULT_BRANCH_TYPE=fix
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

ensure_git_repo() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[ERROR] Not inside a git repository." >&2
    exit 1
  fi
}

ensure_auth() {
  if ! gh auth status >/dev/null 2>&1; then
    echo "[ERROR] gh is not authenticated. Run: gh auth login" >&2
    exit 1
  fi
}

ensure_clean_or_warn() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[WARN] Working tree has uncommitted changes."
    echo "[WARN] Branch switch may fail if changes conflict."
    echo "[WARN] Recommend commit or stash before proceeding."
  fi
}

normalize_branch_type() {
  local branch_type="$1"
  case "$branch_type" in
    fix|feat|chore|refactor|docs|test)
      printf '%s' "$branch_type"
      ;;
    *)
      echo "[ERROR] Unsupported branch type: $branch_type" >&2
      echo "[ERROR] Use one of: fix, feat, chore, refactor, docs, test" >&2
      exit 1
      ;;
  esac
}

slugify() {
  local s="$1"

  s="$(printf '%s' "$s" | tr '[:upper:]' '[:lower:]')"

  # 日本語や記号は落ちるため、ASCII に寄せたタイトルが無い場合は issue を使う
  s="$(printf '%s' "$s" | sed -E 's/[^a-z0-9]+/-/g')"
  s="$(printf '%s' "$s" | sed -E 's/^-+//; s/-+$//; s/-{2,}/-/g')"

  if [[ -z "$s" ]]; then
    s="issue"
  fi

  # 長すぎるブランチ名を避ける
  s="$(printf '%s' "$s" | cut -c1-48)"
  s="$(printf '%s' "$s" | sed -E 's/-+$//')"

  printf '%s' "$s"
}

get_issue_title() {
  local issue_number="$1"
  gh issue view "$issue_number" --json title -q '.title'
}

branch_exists_local() {
  local branch_name="$1"
  git show-ref --verify --quiet "refs/heads/$branch_name"
}

branch_exists_remote() {
  local branch_name="$1"
  git ls-remote --exit-code --heads "$REMOTE_NAME" "$branch_name" >/dev/null 2>&1
}

switch_base_branch() {
  echo "[INFO] Fetching latest refs from $REMOTE_NAME..."
  git fetch "$REMOTE_NAME" --prune

  echo "[INFO] Switching to base branch: $BASE_BRANCH"
  git checkout "$BASE_BRANCH"

  echo "[INFO] Pulling latest $BASE_BRANCH from $REMOTE_NAME..."
  git pull "$REMOTE_NAME" "$BASE_BRANCH"
}

create_branch() {
  local branch_name="$1"
  echo "[INFO] Creating branch: $branch_name"
  git switch -c "$branch_name"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  local issue_number="${1:-}"
  local requested_branch_type="${2:-$DEFAULT_BRANCH_TYPE}"

  if [[ -z "$issue_number" ]]; then
    usage
    exit 1
  fi

  if ! [[ "$issue_number" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] issue_number must be numeric: $issue_number" >&2
    exit 1
  fi

  require_command git
  require_command gh
  ensure_git_repo
  ensure_auth
  ensure_clean_or_warn

  local branch_type
  branch_type="$(normalize_branch_type "$requested_branch_type")"

  echo "[INFO] Reading GitHub Issue #$issue_number ..."
  local issue_title
  issue_title="$(get_issue_title "$issue_number")"

  if [[ -z "$issue_title" ]]; then
    echo "[ERROR] Failed to get title for issue #$issue_number" >&2
    exit 1
  fi

  local slug
  slug="$(slugify "$issue_title")"

  local branch_name="${branch_type}/${issue_number}-${slug}"

  echo "[INFO] Issue title : $issue_title"
  echo "[INFO] Branch type : $branch_type"
  echo "[INFO] Branch name : $branch_name"
  echo "[INFO] Base branch : $BASE_BRANCH"

  switch_base_branch

  if branch_exists_local "$branch_name"; then
    echo "[ERROR] Local branch already exists: $branch_name" >&2
    exit 1
  fi

  if branch_exists_remote "$branch_name"; then
    echo "[ERROR] Remote branch already exists: $branch_name" >&2
    exit 1
  fi

  create_branch "$branch_name"

  echo
  echo "[OK] Ready."
  echo "Current branch: $(git branch --show-current)"
  echo
  echo "Suggested next steps:"
  echo "  git status -sb"
  echo "  # 実装"
  echo "  git add ."
  echo "  git commit -m \"$branch_type: <summary> refs #$issue_number\""
  echo "  git push --set-upstream $REMOTE_NAME $branch_name"
}

main "$@"