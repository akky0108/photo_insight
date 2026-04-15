#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BRANCH_TYPE="${DEFAULT_BRANCH_TYPE:-}"
DEFAULT_LABELS="${DEFAULT_LABELS:-}"
ASSIGNEE="${ASSIGNEE:-@me}"

usage() {
  cat <<'EOF'
GitHub Issue を新規作成し、その Issue から作業ブランチを作成する。

Usage:
  ./scripts/github/issue-new.sh --title "..." [options]

Required:
  --title "..."             Issue タイトル

Optional:
  --type <type>            branch type
                           fix | feat | chore | refactor | docs | test
                           省略時は label から自動判定し、
                           判定不能なら chore

  --body "..."             Issue 本文
  --body-file <path>       Issue 本文ファイル
  --label "a,b,c"          付与するラベル（カンマ区切り）
  --assignee "@me|name"    assignee
                           default: @me
  --no-branch              Issue だけ作成して branch は切らない
  -h, --help               ヘルプ表示

Examples:
  ./scripts/github/issue-new.sh \
    --title "stage 成功判定を厳密化" \
    --label "bug"

  ./scripts/github/issue-new.sh \
    --title "xmp export pipeline を追加" \
    --label "enhancement"

  ./scripts/github/issue-new.sh \
    --title "運用ドキュメント整理" \
    --type docs \
    --body-file docs/tmp_issue.md

Environment:
  DEFAULT_BRANCH_TYPE=
  DEFAULT_LABELS=
  ASSIGNEE=@me
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

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

build_default_body() {
  local title="$1"
  cat <<EOF
## 概要
$title

## 問題
-

## 対応内容
-

## 完了条件
-
EOF
}

read_body_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] body file not found: $path" >&2
    exit 1
  fi
  cat "$path"
}

infer_branch_type_from_labels() {
  local labels_csv="$1"
  local labels_lower

  labels_lower="$(printf '%s' "$labels_csv" | tr '[:upper:]' '[:lower:]')"

  IFS=',' read -r -a labels <<< "$labels_lower"

  local label
  for label in "${labels[@]}"; do
    label="$(trim "$label")"
    case "$label" in
      bug)
        printf '%s' "fix"
        return
        ;;
      enhancement|feature)
        printf '%s' "feat"
        return
        ;;
      documentation|docs)
        printf '%s' "docs"
        return
        ;;
      refactor)
        printf '%s' "refactor"
        return
        ;;
      test|tests)
        printf '%s' "test"
        return
        ;;
    esac
  done

  printf '%s' "chore"
}

create_issue() {
  local title="$1"
  local body="$2"
  local labels_csv="$3"
  local assignee="$4"

  local args=()
  args+=(issue create)
  args+=(--title "$title")
  args+=(--body "$body")

  if [[ -n "$labels_csv" ]]; then
    IFS=',' read -r -a labels <<< "$labels_csv"
    local label
    for label in "${labels[@]}"; do
      label="$(trim "$label")"
      if [[ -n "$label" ]]; then
        args+=(--label "$label")
      fi
    done
  fi

  if [[ -n "$assignee" ]]; then
    args+=(--assignee "$assignee")
  fi

  gh "${args[@]}"
}

extract_issue_number_from_url() {
  local issue_url="$1"
  if [[ "$issue_url" =~ /issues/([0-9]+)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    printf ''
  fi
}

main() {
  local title=""
  local branch_type=""
  local body=""
  local body_file=""
  local labels_csv="$DEFAULT_LABELS"
  local assignee="$ASSIGNEE"
  local no_branch="false"

  if [[ $# -eq 0 ]]; then
    usage
    exit 1
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --title)
        title="${2:-}"
        shift 2
        ;;
      --type)
        branch_type="${2:-}"
        shift 2
        ;;
      --body)
        body="${2:-}"
        shift 2
        ;;
      --body-file)
        body_file="${2:-}"
        shift 2
        ;;
      --label)
        labels_csv="${2:-}"
        shift 2
        ;;
      --assignee)
        assignee="${2:-}"
        shift 2
        ;;
      --no-branch)
        no_branch="true"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "[ERROR] Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  require_command git
  require_command gh
  ensure_git_repo
  ensure_auth

  title="$(trim "$title")"
  if [[ -z "$title" ]]; then
    echo "[ERROR] --title is required." >&2
    exit 1
  fi

  if [[ -n "$body" && -n "$body_file" ]]; then
    echo "[ERROR] Use either --body or --body-file, not both." >&2
    exit 1
  fi

  if [[ -n "$body_file" ]]; then
    body="$(read_body_file "$body_file")"
  fi

  if [[ -z "$body" ]]; then
    body="$(build_default_body "$title")"
  fi

  labels_csv="$(trim "$labels_csv")"

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
  if [[ -n "$labels_csv" ]]; then
    echo "[INFO] Labels      : $labels_csv"
  fi
  if [[ -n "$assignee" ]]; then
    echo "[INFO] Assignee    : $assignee"
  fi

  local issue_url
  issue_url="$(create_issue "$title" "$body" "$labels_csv" "$assignee")"

  if [[ -z "$issue_url" ]]; then
    echo "[ERROR] Failed to create issue." >&2
    exit 1
  fi

  local issue_number
  issue_number="$(extract_issue_number_from_url "$issue_url")"

  if [[ -z "$issue_number" ]]; then
    echo "[ERROR] Failed to parse issue number from URL: $issue_url" >&2
    exit 1
  fi

  echo "[OK] Issue created: #$issue_number"
  echo "URL: $issue_url"

  if [[ "$no_branch" == "true" ]]; then
    echo "[INFO] --no-branch specified. Skipping branch creation."
    exit 0
  fi

  echo "[INFO] Starting branch from issue number: $issue_number"
  ./scripts/github/start_issue.sh "$issue_number" "$branch_type"
}

main "$@"