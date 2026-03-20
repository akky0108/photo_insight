#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/photo_insight"

cd "${REPO}"

echo "========================================"
echo " photo_insight PREPARE START"
echo "========================================"

echo "[STEP 1] config deploy"
bash scripts/deploy/deploy_config.sh

echo "[STEP 2] image deploy"
bash scripts/deploy/deploy_prod.sh

echo "========================================"
echo " PREPARE DONE"
echo "========================================"
