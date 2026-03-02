#!/usr/bin/env bash
set -euo pipefail

# 例: ./scripts/release_prod_image.sh prod-20260302
TAG="${1:-prod-$(date +%Y%m%d)}"

cd "$(dirname "$0")/.."

echo "[INFO] building prod image tag: photo_insight:${TAG}"
PHOTO_INSIGHT_IMAGE_TAG="${TAG}" docker compose -f compose.release.yaml build --no-cache

echo "[INFO] built image:"
docker images "photo_insight:${TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}"

echo "[NEXT] set ~/photo_insight_prod/config/.env PHOTO_INSIGHT_IMAGE_TAG=${TAG}"