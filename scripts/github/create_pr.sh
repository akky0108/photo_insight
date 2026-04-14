#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${BASE_BRANCH:-develop}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
DRAFT_MODE="${DRAFT_MODE:-false}"

usage() {
  cat <<'EOF'
現在の作業ブランチから GitHub Pull Request を作成する。

Usage:
  ./scripts/github/create_pr.sh [--draft]

Examples:
  ./scripts/github/create_pr.sh
  ./scripts/github/create_pr.sh --draft

Environment variables:
  BASE_BRANCH=develop
  REMOTE_NAME=origin
  DRAFT_MODE=false
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

current_branch() {
  git branch --show-current
}

ensure_not_base_branch() {
  local branch="$1"
  if [[ "$branch" == "$BASE_BRANCH" ]]; then
    echo "[ERROR] Current branch is base branch: $BASE_BRANCH" >&2
    echo "[ERROR] Switch to a work branch before creating a PR." >&2
    exit 1
  fi
}

ensure_has_commits_against_base() {
  local branch="$1"
  local count
  count="$(git rev-list --count "${REMOTE_NAME}/${BASE_BRANCH}..${branch}" 2>/dev/null || echo 0)"
  if [[ "$count" == "0" ]]; then
    echo "[WARN] No commits found on $branch compared to ${REMOTE_NAME}/${BASE_BRANCH}."
    echo "[WARN] PR may be empty."
  fi
}

ensure_branch_has_upstream_or_push() {
  local branch="$1"

  if git rev-parse --abbrev-ref "${branch}@{upstream}" >/dev/null 2>&1; then
    return 0
  fi

  echo "[INFO] No upstream set for $branch. Pushing with upstream..."
  git push --set-upstream "$REMOTE_NAME" "$branch"
}

extract_issue_number() {
  local branch="$1"
  if [[ "$branch" =~ ^[^/]+/([0-9]+)- ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    printf ''
  fi
}

extract_branch_type() {
  local branch="$1"
  if [[ "$branch" =~ ^([^/]+)/ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    printf 'chore'
  fi
}

get_issue_title() {
  local issue_number="$1"
  gh issue view "$issue_number" --json title -q '.title'
}

capitalize_first() {
  local s="$1"
  if [[ -z "$s" ]]; then
    printf '%s' "$s"
    return
  fi
  printf '%s%s' "$(printf '%s' "${s:0:1}" | tr '[:lower:]' '[:upper:]')" "${s:1}"
}

build_pr_title() {
  local branch_type="$1"
  local issue_number="$2"
  local issue_title="$3"

  local title_prefix
  title_prefix="$(capitalize_first "$branch_type")"

  if [[ -n "$issue_number" && -n "$issue_title" ]]; then
    printf '%s: %s (#%s)' "$title_prefix" "$issue_title" "$issue_number"
  elif [[ -n "$issue_title" ]]; then
    printf '%s: %s' "$title_prefix" "$issue_title"
  else
    printf '%s: update' "$title_prefix"
  fi
}

build_pr_body() {
  local issue_number="$1"
  local issue_title="$2"
  local branch="$3"

  cat <<EOF
## 概要
- ${issue_title:-関連Issue対応}
- branch: \`${branch}\`

## 対応内容
- 実装内容を記載
- テスト内容を記載
- 必要ならドキュメント更新を記載

## 確認
- [ ] ローカルで動作確認
- [ ] pytest 実行
- [ ] 影響範囲確認

## 関連
$( [[ -n "$issue_number" ]] && printf '%s\n' "Closes #${issue_number}" || printf '%s\n' "- 関連Issueなし" )
EOF
}

ensure_no_existing_pr() {
  local branch="$1"
  local existing
  existing="$(gh pr list --head "$branch" --base "$BASE_BRANCH" --json number -q '.[0].number' 2>/dev/null || true)"
  if [[ -n "$existing" ]]; then
    echo "[ERROR] PR already exists for branch '$branch' -> '$BASE_BRANCH': #$existing" >&2
    exit 1
  fi
}

main() {
  local draft_flag="$DRAFT_MODE"

  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  if [[ "${1:-}" == "--draft" ]]; then
    draft_flag="true"
  elif [[ $# -gt 0 ]]; then
    echo "[ERROR] Unknown argument: $1" >&2
    usage
    exit 1
  fi

  require_command git
  require_command gh
  ensure_git_repo
  ensure_auth

  echo "[INFO] Fetching latest refs from $REMOTE_NAME..."
  git fetch "$REMOTE_NAME" --prune

  local branch
  branch="$(current_branch)"

  if [[ -z "$branch" ]]; then
    echo "[ERROR] Could not determine current branch." >&2
    exit 1
  fi

  ensure_not_base_branch "$branch"
  ensure_branch_has_upstream_or_push "$branch"
  ensure_has_commits_against_base "$branch"
  ensure_no_existing_pr "$branch"

  local issue_number
  issue_number="$(extract_issue_number "$branch")"

  local branch_type
  branch_type="$(extract_branch_type "$branch")"

  local issue_title=""
  if [[ -n "$issue_number" ]]; then
    echo "[INFO] Reading GitHub Issue #$issue_number ..."
    issue_title="$(get_issue_title "$issue_number" || true)"
  fi

  local pr_title
  pr_title="$(build_pr_title "$branch_type" "$issue_number" "$issue_title")"

  local pr_body
  pr_body="$(build_pr_body "$issue_number" "$issue_title" "$branch")"

  echo "[INFO] Base branch : $BASE_BRANCH"
  echo "[INFO] Head branch : $branch"
  echo "[INFO] PR title    : $pr_title"

  if [[ "$draft_flag" == "true" ]]; then
    echo "[INFO] Creating draft PR..."
    gh pr create \
      --base "$BASE_BRANCH" \
      --head "$branch" \
      --title "$pr_title" \
      --body "$pr_body" \
      --draft
  else
    echo "[INFO] Creating PR..."
    gh pr create \
      --base "$BASE_BRANCH" \
      --head "$branch" \
      --title "$pr_title" \
      --body "$pr_body"
  fi

  echo
  echo "[OK] PR created."
  echo "Next:"
  echo "  - GitHub Web で base=head を最終確認"
  echo "  - CI を確認"
  echo "  - Review 後に develop へマージ"
}

main "$@"