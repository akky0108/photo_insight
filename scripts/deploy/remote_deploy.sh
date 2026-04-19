#!/usr/bin/env bash
set -euo pipefail

PROD_APP_DIR="${PROD_APP_DIR:-$HOME/photo_insight_prod}"
PROD_ENV_FILE="${PROD_ENV_FILE:-config/.env}"
GIT_SHA="${GIT_SHA:-unknown}"
GIT_REF_NAME="${GIT_REF_NAME:-main}"
GITHUB_RUN_ID="${GITHUB_RUN_ID:-unknown}"
GITHUB_ACTOR="${GITHUB_ACTOR:-unknown}"

log_info() {
  echo "[INFO] $*"
}

log_error() {
  echo "[ERROR] $*" >&2
}

die() {
  log_error "$*"
  exit 1
}

require_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Required file not found: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Required directory not found: $path"
}

ensure_safe_prod_dir() {
  [[ -n "$PROD_APP_DIR" ]] || die "PROD_APP_DIR is empty"
  [[ "$PROD_APP_DIR" != "/" ]] || die "PROD_APP_DIR must not be /"
}

main() {
  log_info "Starting remote production deploy"
  log_info "PROD_APP_DIR=$PROD_APP_DIR"
  log_info "PROD_ENV_FILE=$PROD_ENV_FILE"
  log_info "GIT_REF_NAME=$GIT_REF_NAME"
  log_info "GIT_SHA=$GIT_SHA"
  log_info "GITHUB_RUN_ID=$GITHUB_RUN_ID"
  log_info "GITHUB_ACTOR=$GITHUB_ACTOR"

  ensure_safe_prod_dir
  require_command docker
  require_dir "$PROD_APP_DIR"

  cd "$PROD_APP_DIR"

  require_file "compose.prod.yaml"
  require_file "$PROD_ENV_FILE"

  log_info "Validating docker compose configuration"
  docker compose -f compose.prod.yaml --env-file "$PROD_ENV_FILE" config -q

  log_info "Showing configured services"
  docker compose -f compose.prod.yaml --env-file "$PROD_ENV_FILE" config --services

  log_info "Showing compose images"
  docker compose -f compose.prod.yaml --env-file "$PROD_ENV_FILE" images || true

  log_info "Updating RELEASE.txt"
  cat > RELEASE.txt <<EOF
deployed_at=$(date '+%Y-%m-%d %H:%M:%S %z')
git_ref_name=$GIT_REF_NAME
git_sha=$GIT_SHA
github_run_id=$GITHUB_RUN_ID
github_actor=$GITHUB_ACTOR
EOF

  log_info "Remote production deploy completed successfully"
}

main "$@"