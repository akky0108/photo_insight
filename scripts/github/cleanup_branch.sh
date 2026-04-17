#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${BASE_BRANCH:-develop}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
DELETE_REMOTE="${DELETE_REMOTE:-true}"

usage() {
  cat <<'EOF'
マージ済みの作業ブランチを cleanup する。

Usage:
  ./scripts/github/cleanup_branch.sh [branch_name] [--keep-remote]

Arguments:
  branch_name     削除対象ブランチ名
                  省略時は「現在のブランチ」を対象にする

Options:
  --keep-remote   リモートブランチは削除しない

Examples:
  ./scripts/github/cleanup_branch.sh
  ./scripts/github/cleanup_branch.sh fix/123-stage-success-criteria
  ./scripts/github/cleanup_branch.sh fix/123-stage-success-criteria --keep-remote

Environment variables:
  BASE_BRANCH=develop
  REMOTE_NAME=origin
  DELETE_REMOTE=true
EOF
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $cmd" >&2
    exit 1
  fi
}

has_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1
}

ensure_git_repo() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[ERROR] Not inside a git repository." >&2
    exit 1
  fi
}

current_branch() {
  git branch --show-current
}

branch_exists_local() {
  local branch_name="$1"
  git show-ref --verify --quiet "refs/heads/$branch_name"
}

branch_exists_remote() {
  local branch_name="$1"
  git ls-remote --exit-code --heads "$REMOTE_NAME" "$branch_name" >/dev/null 2>&1
}

ensure_safe_target() {
  local branch_name="$1"

  if [[ -z "$branch_name" ]]; then
    echo "[ERROR] Target branch is empty." >&2
    exit 1
  fi

  if [[ "$branch_name" == "$BASE_BRANCH" ]]; then
    echo "[ERROR] Refusing to delete base branch: $BASE_BRANCH" >&2
    exit 1
  fi

  if [[ "$branch_name" == "main" ]]; then
    echo "[ERROR] Refusing to delete protected branch: main" >&2
    exit 1
  fi

  if [[ "$branch_name" == "master" ]]; then
    echo "[ERROR] Refusing to delete protected branch: master" >&2
    exit 1
  fi
}

ensure_clean_or_warn() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[WARN] Working tree has uncommitted changes."
    echo "[WARN] Checkout to $BASE_BRANCH may fail if changes conflict."
  fi
}

switch_to_base_branch() {
  echo "[INFO] Fetching latest refs from $REMOTE_NAME..."
  git fetch "$REMOTE_NAME" --prune

  echo "[INFO] Switching to base branch: $BASE_BRANCH"
  git checkout "$BASE_BRANCH"

  echo "[INFO] Pulling latest $BASE_BRANCH from $REMOTE_NAME..."
  git pull "$REMOTE_NAME" "$BASE_BRANCH"
}

is_merged_into_base() {
  local branch_name="$1"
  git merge-base --is-ancestor "$branch_name" "$BASE_BRANCH"
}

gh_is_available_and_authenticated() {
  if ! has_command gh; then
    return 1
  fi

  gh auth status >/dev/null 2>&1
}

find_merged_pr_number() {
  local branch_name="$1"

  gh pr list \
    --state merged \
    --base "$BASE_BRANCH" \
    --head "$branch_name" \
    --limit 1 \
    --json number \
    --jq '.[0].number // ""'
}

is_merged_via_github_pr() {
  local branch_name="$1"

  if ! gh_is_available_and_authenticated; then
    return 1
  fi

  local pr_number=""
  pr_number="$(find_merged_pr_number "$branch_name" || true)"

  [[ -n "$pr_number" ]]
}

print_github_merge_hint() {
  local branch_name="$1"

  if ! gh_is_available_and_authenticated; then
    echo "[INFO] gh is not available or not authenticated. GitHub merged PR check skipped."
    return 0
  fi

  local pr_info=""
  pr_info="$(
    gh pr list \
      --state merged \
      --base "$BASE_BRANCH" \
      --head "$branch_name" \
      --limit 1 \
      --json number,title,url,mergedAt \
      --jq '.[0] | "PR #\(.number) merged at \(.mergedAt): \(.title) (\(.url))"' \
      2>/dev/null || true
  )"

  if [[ -n "$pr_info" ]]; then
    echo "[INFO] $pr_info"
  fi
}

delete_local_branch() {
  local branch_name="$1"
  echo "[INFO] Deleting local branch: $branch_name"
  git branch -d "$branch_name"
}

force_delete_local_branch() {
  local branch_name="$1"
  echo "[WARN] Force deleting local branch: $branch_name"
  git branch -D "$branch_name"
}

delete_remote_branch() {
  local branch_name="$1"
  echo "[INFO] Deleting remote branch: $REMOTE_NAME/$branch_name"
  git push "$REMOTE_NAME" --delete "$branch_name"
}

main() {
  local target_branch=""
  local delete_remote="$DELETE_REMOTE"

  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  for arg in "$@"; do
    case "$arg" in
      --keep-remote)
        delete_remote="false"
        ;;
      -*)
        echo "[ERROR] Unknown option: $arg" >&2
        usage
        exit 1
        ;;
      *)
        if [[ -z "$target_branch" ]]; then
          target_branch="$arg"
        else
          echo "[ERROR] Too many positional arguments." >&2
          usage
          exit 1
        fi
        ;;
    esac
  done

  require_command git
  ensure_git_repo
  ensure_clean_or_warn

  if [[ -z "$target_branch" ]]; then
    target_branch="$(current_branch)"
  fi

  ensure_safe_target "$target_branch"

  echo "[INFO] Target branch : $target_branch"
  echo "[INFO] Base branch   : $BASE_BRANCH"
  echo "[INFO] Remote delete : $delete_remote"

  if ! branch_exists_local "$target_branch"; then
    echo "[WARN] Local branch does not exist: $target_branch"
  fi

  switch_to_base_branch

  if branch_exists_local "$target_branch"; then
    if is_merged_into_base "$target_branch"; then
      echo "[INFO] Branch is merged into $BASE_BRANCH by git ancestry."
      delete_local_branch "$target_branch"
    elif is_merged_via_github_pr "$target_branch"; then
      echo "[INFO] Branch is not merged by git ancestry, but GitHub shows a merged PR."
      echo "[INFO] Treating this as squash/rebase merged and proceeding."
      print_github_merge_hint "$target_branch"
      force_delete_local_branch "$target_branch"
    else
      echo "[WARN] Branch is not merged into $BASE_BRANCH: $target_branch"
      echo "[WARN] No merged GitHub PR was found for this branch."
      echo "[WARN] Refusing delete."
      echo "[WARN] If you really want to discard it, run:"
      echo "       git branch -D \"$target_branch\""
      exit 1
    fi
  fi

  if [[ "$delete_remote" == "true" ]]; then
    if branch_exists_remote "$target_branch"; then
      if is_merged_via_github_pr "$target_branch" || ! branch_exists_local "$target_branch"; then
        delete_remote_branch "$target_branch"
      else
        echo "[WARN] Remote branch still exists, but no merged PR was confirmed."
        echo "[WARN] Refusing remote delete."
        exit 1
      fi
    else
      echo "[INFO] Remote branch does not exist: $REMOTE_NAME/$target_branch"
    fi
  fi

  echo
  echo "[OK] Cleanup completed."
  echo "Current branch: $(git branch --show-current)"
}

main "$@"