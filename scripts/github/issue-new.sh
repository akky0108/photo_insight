#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BRANCH_TYPE="${DEFAULT_BRANCH_TYPE:-}"
DEFAULT_LABELS="${DEFAULT_LABELS:-}"
ASSIGNEE="${ASSIGNEE:-@me}"

usage() {
  cat <<'EOF'
GitHub Issue を新規作成し、その Issue から作業ブランチを作成する。
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] Required command not found: $1" >&2
    exit 1
  }
}

ensure_git_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "[ERROR] Not inside a git repository." >&2
    exit 1
  }
}

ensure_auth() {
  gh auth status >/dev/null 2>&1 || {
    echo "[ERROR] gh is not authenticated." >&2
    exit 1
  }
}

normalize_branch_type() {
  case "$1" in
    fix|feat|chore|refactor|docs|test) echo "$1" ;;
    *) echo "[ERROR] Unsupported branch type: $1" >&2; exit 1 ;;
  esac
}

trim() {
  local s="$1"
  # leading
  s="${s#"${s%%[![:space:]]*}"}"
  # trailing
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

infer_branch_type_from_labels() {
  local labels_csv="$1"
  local labels_lower
  labels_lower="$(printf '%s' "$labels_csv" | tr '[:upper:]' '[:lower:]')"

  IFS=',' read -r -a labels <<< "$labels_lower"

  for label in "${labels[@]}"; do
    label="$(trim "$label")"
    case "$label" in
      bug) echo "fix"; return ;;
      enhancement|feature) echo "feat"; return ;;
      documentation|docs) echo "docs"; return ;;
      refactor) echo "refactor"; return ;;
      test|tests) echo "test"; return ;;
    esac
  done

  echo "chore"
}

main() {
  local title=""
  local branch_type="$DEFAULT_BRANCH_TYPE"
  local body=""
  local labels_csv="$DEFAULT_LABELS"
  local assignee="$ASSIGNEE"
  local no_branch="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --title) title="$2"; shift 2 ;;
      --type) branch_type="$2"; shift 2 ;;
      --label) labels_csv="$2"; shift 2 ;;
      --no-branch) no_branch="true"; shift ;;
      *) shift ;;
    esac
  done

  require_command git
  require_command gh
  ensure_git_repo
  ensure_auth

  title="$(trim "$title")"

  labels_csv="$(trim "$labels_csv")"

  # ★ 統一ロジック
  if [[ -n "$branch_type" ]]; then
    branch_type="$(normalize_branch_type "$branch_type")"
    echo "[INFO] Branch type explicitly specified: $branch_type"
  elif [[ -n "$DEFAULT_BRANCH_TYPE" ]]; then
    branch_type="$(normalize_branch_type "$DEFAULT_BRANCH_TYPE")"
    echo "[INFO] Branch type from DEFAULT_BRANCH_TYPE: $branch_type"
  else
    branch_type="$(infer_branch_type_from_labels "$labels_csv")"
    echo "[INFO] Branch type inferred from labels: $branch_type"
  fi

  echo "[INFO] Creating GitHub Issue..."
  echo "[INFO] Title       : $title"
  echo "[INFO] Branch type : $branch_type"

  local issue_url
  issue_url="$(gh issue create --title "$title")"

  local issue_number
  issue_number="$(echo "$issue_url" | grep -o '[0-9]\+$')"

  echo "[OK] Issue created: #$issue_number"

  if [[ "$no_branch" == "true" ]]; then
    exit 0
  fi

  echo "[INFO] Creating work branch from issue #$issue_number ..."
  ./scripts/github/start_issue.sh "$issue_number" "$branch_type"
}

main "$@"