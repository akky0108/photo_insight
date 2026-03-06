#!/usr/bin/env bash
set -euo pipefail

PROD="${HOME}/photo_insight_prod"
ENV_FILE="${PROD}/config/.env"
COMPOSE_FILE="${PROD}/compose.prod.yaml"

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "[ERROR] compose not found: ${COMPOSE_FILE}" >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERROR] env not found: ${ENV_FILE}" >&2
  exit 1
fi

# Always run from PROD dir so relative paths in compose work (./config, ./runs, etc.)
cd "${PROD}"

# Ensure project_root points to /work (where runs/, config/, input/ are mounted)
exec env PROJECT_ROOT=/work docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"