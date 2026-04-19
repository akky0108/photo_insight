#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BRANCH_TYPE="${DEFAULT_BRANCH_TYPE:-}"
DEFAULT_LABELS="${DEFAULT_LABELS:-}"
ASSIGNEE="${ASSIGNEE:-@me}"

trim() {
  local s="$1"
  # leading
  s="${s#"${s%%[![:space:]]*}"}"
  # trailing
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

normalize_branch_type() {
  local branch_type="$1"
  case "$branch_type" in
    fix|feat|chore|refactor|docs|test)
      printf '%s' "$branch_type"
      ;;
    *)
      echo "[ERROR] Unsupported branch type: $branch_type" >&2
      exit 1
      ;;
  esac
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
      bug) echo "fix"; return ;;
      enhancement|feature) echo "feat"; return ;;
      docs|documentation) echo "docs"; return ;;
      refactor) echo "refactor"; return ;;
      test|tests) echo "test"; return ;;
    esac
  done

  echo "chore"
}

create_issue() {
  local title="$1"
  local body="$2"
  local labels_csv="$3"
  local assignee="$4"

  local args=()
  args+=(issue create --title "$title" --body "$body")

  if [[ -n "$labels_csv" ]]; then
    IFS=',' read -r -a labels <<< "$labels_csv"
    for label in "${labels[@]}"; do
      label="$(trim "$label")"
      [[ -n "$label" ]] && args+=(--label "$label")
    done
  fi

  [[ -n "$assignee" ]] && args+=(--assignee "$assignee")

  gh "${args[@]}"
}

extract_issue_number_from_url() {
  local issue_url="$1"
  if [[ "$issue_url" =~ /issues/([0-9]+)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  fi
}

main() {
  local title=""
  local branch_type=""
  local body=""
  local labels_csv="$DEFAULT_LABELS"
  local assignee="$ASSIGNEE"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --title) title="$2"; shift 2 ;;
      --type) branch_type="$2"; shift 2 ;;
      --label) labels_csv="$2"; shift 2 ;;
      --assignee) assignee="$2"; shift 2 ;;
      *) shift ;;
    esac
  done

  title="$(trim "$title")"
  [[ -z "$title" ]] && { echo "[ERROR] title required"; exit 1; }

  labels_csv="$(trim "$labels_csv")"

  if [[ -n "$branch_type" ]]; then
    branch_type="$(normalize_branch_type "$branch_type")"
  else
    branch_type="$(infer_branch_type_from_labels "$labels_csv")"
  fi

  echo "[INFO] Creating Issue..."
  issue_url="$(create_issue "$title" "$body" "$labels_csv" "$assignee")"

  issue_number="$(extract_issue_number_from_url "$issue_url")"

  echo "[OK] Issue #$issue_number created"

  ./scripts/github/start_issue.sh "$issue_number" "$branch_type"
}

main "$@"