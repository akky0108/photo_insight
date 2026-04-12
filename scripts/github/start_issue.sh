#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${BASE_BRANCH:-develop}"
BRANCH_TYPE="${BRANCH_TYPE:-fix}"
REMOTE_NAME="${REMOTE_NAME:-origin}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/github/start_issue.sh <issue_number> [branch_type]

Examples:
  ./scripts/github/start_issue.sh 123
  ./scripts/github/start_issue.sh 123 feat
  BRANCH_TYPE=chore ./scripts/github/start_issue.sh 123

Notes:
  - default base branch: develop
  - default branch type: fix
  - requires: gh, git
EOF
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $cmd" >&2
    exit 1
  fi
}

slugify() {
  local s="$1"

  s="$(printf '%s' "$s" | tr '[:upper:]' '[:lower:]')"
  s="$(printf '%s' "$s" | sed -E 's/[^a-z0-9]+/-/g')"
  s="$(printf '%s' "$s" | sed -E 's/^-+//; s/-+$//; s/-{2,}/-/g')"

  if [[ -z "$s" ]]; then
    s="issue"
  fi

  printf '%s' "$s"
}

ensure_clean_or_warn() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[WARN] Working tree has uncommitted changes."
    echo "       Commit or stash them first if branch switch may conflict."
  fi
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  local issue_number="${1:-}"
  local branch_type="${2:-$BRANCH_TYPE}"

  if [[ -z "$issue_number" ]]; then
    usage
    exit 1
  fi

  if ! [[ "$issue_number" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] issue_number must be numeric: $issue_number" >&2
    exit 1
  fi

  case "$branch_type" in
    fix|feat|chore|refactor|docs|test)
      ;;
    *)
      echo "[ERROR] Unsupported branch type: $branch_type" >&2
      echo "        Use one of: fix, feat, chore, refactor, docs, test" >&2
      exit 1
      ;;
  esac

  require_command git
  require_command gh

  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "[ERROR] Not inside a git repository." >&2
    exit 1
  }

  ensure_clean_or_warn

  echo "[INFO] Fetching issue #$issue_number title..."
  local issue_title
  issue_title="$(gh issue view "$issue_number" --json title -q '.title')"

  if [[ -z "$issue_title" ]]; then
    echo "[ERROR] Could not retrieve title for issue #$issue_number" >&2
    exit 1
  fi

  local slug
  slug="$(slugify "$issue_title")"

  local branch_name="${branch_type}/${issue_number}-${slug}"

  echo "[INFO] Issue title : $issue_title"
  echo "[INFO] Branch name : $branch_name"
  echo "[INFO] Base branch : $BASE_BRANCH"

  echo "[INFO] Fetching remote..."
  git fetch "$REMOTE_NAME" --prune

  echo "[INFO] Switching to $BASE_BRANCH..."
  git checkout "$BASE_BRANCH"

  echo "[INFO] Pulling latest $BASE_BRANCH from $REMOTE_NAME..."
  git pull "$REMOTE_NAME" "$BASE_BRANCH"

  if git show-ref --verify --quiet "refs/heads/$branch_name"; then
    echo "[ERROR] Local branch already exists: $branch_name" >&2
    exit 1
  fi

  if git ls-remote --exit-code --heads "$REMOTE_NAME" "$branch_name" >/dev/null 2>&1; then
    echo "[ERROR] Remote branch already exists: $branch_name" >&2
    exit 1
  fi

  echo "[INFO] Creating branch..."
  git switch -c "$branch_name"

  echo
  echo "[OK] Ready to start work."
  echo "Current branch: $(git branch --show-current)"
  echo
  echo "Next steps:"
  echo "  1) implement changes"
  echo "  2) git status -sb"
  echo "  3) git add ."
  echo "  4) git commit -m \"$branch_type: <summary> refs #$issue_number\""
  echo "  5) git push --set-upstream $REMOTE_NAME $branch_name"
}

main "$@"