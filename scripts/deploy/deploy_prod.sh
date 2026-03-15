#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/photo_insight"
PROD="${HOME}/photo_insight_prod"

# 本番用 Dockerfile
PROD_DOCKERFILE="docker/Dockerfile.prod"
PROD_DOCKER_TARGET="runtime"

RUN_AFTER=false
NO_CACHE=false
TAG=""
KEEP=5
CLEAN_IMAGES=true
WRITE_RELEASE=true
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  deploy_prod.sh [options]

Options:
  --tag TAG            Use specific image tag (e.g. prod-20260302 or prod-20260302_120000)
  --run                Run production container after deploy
  --no-cache           Build image with --no-cache
  --keep N             Keep latest N images matching photo_insight:prod-* (default: 5)
  --no-clean           Do not remove old prod-* images
  --no-release         Do not write RELEASE.txt
  --dry-run            Do not remove images / do not run container (build+deploy+env update+release write only)
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_AFTER=true; shift ;;
    --no-cache) NO_CACHE=true; shift ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    --keep) KEEP="${2:-}"; shift 2 ;;
    --no-clean) CLEAN_IMAGES=false; shift ;;
    --no-release) WRITE_RELEASE=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TAG}" ]]; then
  TAG="prod-$(date +%Y%m%d_%H%M%S)"
fi

if [[ ! "${KEEP}" =~ ^[0-9]+$ ]] || [[ "${KEEP}" -lt 1 ]]; then
  echo "[ERROR] --keep must be a positive integer. got: ${KEEP}" >&2
  exit 2
fi

if [[ ! -d "${REPO}" ]]; then
  echo "[ERROR] repo not found: ${REPO}" >&2
  exit 1
fi

TEMPLATE_COMPOSE="${REPO}/deploy/prod/compose.prod.yaml"
TEMPLATE_ENV="${REPO}/deploy/prod/env.example"
RUNNER_SCRIPT="${REPO}/scripts/ops/run_prod.sh"
DOCKERFILE_PATH="${REPO}/${PROD_DOCKERFILE}"

if [[ ! -f "${TEMPLATE_COMPOSE}" ]]; then
  echo "[ERROR] missing template: ${TEMPLATE_COMPOSE}" >&2
  exit 1
fi

if [[ ! -f "${TEMPLATE_ENV}" ]]; then
  echo "[ERROR] missing template: ${TEMPLATE_ENV}" >&2
  exit 1
fi

if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
  echo "[ERROR] missing runner script (expected by deploy): ${RUNNER_SCRIPT}" >&2
  echo "        Create it in repo (scripts/ops/run_prod.sh) and re-run." >&2
  exit 1
fi

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
  echo "[ERROR] missing prod dockerfile: ${DOCKERFILE_PATH}" >&2
  exit 1
fi

mkdir -p "${PROD}/"{config,logs,runs,tmp,output}

echo "[INFO] repo              : ${REPO}"
echo "[INFO] prod              : ${PROD}"
echo "[INFO] prod dockerfile   : ${PROD_DOCKERFILE}"
echo "[INFO] prod target       : ${PROD_DOCKER_TARGET}"
echo "[INFO] tag               : ${TAG}"
echo "[INFO] keep              : ${KEEP}"
echo "[INFO] flags             : run=${RUN_AFTER} no_cache=${NO_CACHE} clean=${CLEAN_IMAGES} release=${WRITE_RELEASE} dry_run=${DRY_RUN}"

set_kv() {
  local key="$1"
  local value="$2"
  local file="$3"

  if grep -qE "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file"
  else
    printf "\n%s=%s\n" "$key" "$value" >> "$file"
  fi
}

ensure_kv() {
  local key="$1"
  local value="$2"
  local file="$3"

  if ! grep -qE "^${key}=" "$file"; then
    printf "\n%s=%s\n" "$key" "$value" >> "$file"
  fi
}

# ---------------------------------------------------------------------
# 1) Build prod image (repo side)
# ---------------------------------------------------------------------
cd "${REPO}"

GIT_SHA="unknown"
GIT_BRANCH="unknown"
if command -v git >/dev/null 2>&1; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  fi
fi

BUILD_CMD=(
  docker build
  -t "photo_insight:${TAG}"
  -f "${PROD_DOCKERFILE}"
  --target "${PROD_DOCKER_TARGET}"
  .
)

if [[ "${NO_CACHE}" == "true" ]]; then
  BUILD_CMD=(docker build --no-cache -t "photo_insight:${TAG}" -f "${PROD_DOCKERFILE}" --target "${PROD_DOCKER_TARGET}" .)
fi

echo "[INFO] building production image..."
printf '[INFO] command:'
printf ' %q' "${BUILD_CMD[@]}"
printf '\n'

"${BUILD_CMD[@]}"

IMAGE_ID="$(docker images -q "photo_insight:${TAG}" | head -n 1 || true)"
echo "[INFO] built image: photo_insight:${TAG} (id=${IMAGE_ID:-unknown})"

# ---------------------------------------------------------------------
# 2) Deploy compose + env template + runner to PROD
# ---------------------------------------------------------------------
install -m 0644 "${TEMPLATE_COMPOSE}" "${PROD}/compose.prod.yaml"

if [[ ! -f "${PROD}/config/.env" ]]; then
  install -m 0600 "${TEMPLATE_ENV}" "${PROD}/config/.env"
  echo "[INFO] created ${PROD}/config/.env from env.example"
fi

install -m 0755 "${RUNNER_SCRIPT}" "${PROD}/run_prod.sh"
echo "[INFO] installed runner: ${PROD}/run_prod.sh"

ENV_FILE="${PROD}/config/.env"

# ---------------------------------------------------------------------
# 3) Update PROD config/.env safely
# ---------------------------------------------------------------------
set_kv "PHOTO_INSIGHT_IMAGE_TAG" "${TAG}" "${ENV_FILE}"
ensure_kv "PHOTO_INSIGHT_INPUT_DIR" "/mnt/l/picture" "${ENV_FILE}"
ensure_kv "UID" "$(id -u)" "${ENV_FILE}"
ensure_kv "GID" "$(id -g)" "${ENV_FILE}"

echo "[INFO] prod .env summary:"
grep -E '^(PHOTO_INSIGHT_IMAGE_TAG|PHOTO_INSIGHT_INPUT_DIR|UID|GID)=' "${ENV_FILE}" || true

# ---------------------------------------------------------------------
# 4) Write RELEASE.txt (prod side) for traceability
# ---------------------------------------------------------------------
if [[ "${WRITE_RELEASE}" == "true" ]]; then
  RELEASE_FILE="${PROD}/RELEASE.txt"
  NOW="$(date -Iseconds)"
  INPUT_DIR="$(grep -E '^PHOTO_INSIGHT_INPUT_DIR=' "${ENV_FILE}" | head -n 1 | cut -d= -f2- || true)"
  {
    echo "timestamp=${NOW}"
    echo "image=photo_insight:${TAG}"
    echo "image_id=${IMAGE_ID:-unknown}"
    echo "repo=${REPO}"
    echo "git_branch=${GIT_BRANCH}"
    echo "git_sha=${GIT_SHA}"
    echo "prod_dir=${PROD}"
    echo "input_dir=${INPUT_DIR}"
    echo "dockerfile=${PROD_DOCKERFILE}"
    echo "target=${PROD_DOCKER_TARGET}"
  } > "${RELEASE_FILE}"
  echo "[INFO] wrote ${RELEASE_FILE}"
fi

# ---------------------------------------------------------------------
# 5) Cleanup old images (keep latest N prod-*)
#   - sort by TAG (prod-YYYYMMDD_HHMMSS) desc for stability
# ---------------------------------------------------------------------
if [[ "${CLEAN_IMAGES}" == "true" ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] skip image cleanup"
  else
    echo "[INFO] cleaning old images: keep latest ${KEEP} of photo_insight:prod-*"

    mapfile -t LINES < <(
      docker images --format '{{.Repository}} {{.Tag}} {{.ID}}' photo_insight \
        | awk '$2 ~ /^prod-/ {print $0}' \
        | sort -r -k2,2
    )

    if [[ "${#LINES[@]}" -le "${KEEP}" ]]; then
      echo "[INFO] nothing to clean (found ${#LINES[@]} prod-* images)"
    else
      echo "[INFO] found ${#LINES[@]} prod-* images"
      REMOVABLE_IDS=()
      idx=0
      for line in "${LINES[@]}"; do
        idx=$((idx+1))
        repo="$(echo "$line" | awk '{print $1}')"
        tag="$(echo "$line"  | awk '{print $2}')"
        id="$(echo "$line"   | awk '{print $3}')"
        if [[ "${idx}" -le "${KEEP}" ]]; then
          echo "[KEEP] ${repo}:${tag} ${id}"
        else
          echo "[RM? ] ${repo}:${tag} ${id}"
          REMOVABLE_IDS+=("${id}")
        fi
      done

      UNIQUE_IDS=()
      for id in "${REMOVABLE_IDS[@]}"; do
        seen=false
        for u in "${UNIQUE_IDS[@]:-}"; do
          if [[ "$u" == "$id" ]]; then
            seen=true
            break
          fi
        done
        if [[ "$seen" == "false" ]]; then
          UNIQUE_IDS+=("$id")
        fi
      done

      if [[ "${#UNIQUE_IDS[@]}" -gt 0 ]]; then
        echo "[INFO] removing ${#UNIQUE_IDS[@]} image IDs..."
        for id in "${UNIQUE_IDS[@]}"; do
          docker rmi "${id}" >/dev/null 2>&1 || echo "[WARN] could not remove image id=${id} (maybe in use)"
        done
        echo "[INFO] cleanup done"
      fi
    fi
  fi
fi

# ---------------------------------------------------------------------
# 6) Optional run (use runner to ensure --env-file is applied)
# ---------------------------------------------------------------------
if [[ "${RUN_AFTER}" == "true" ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] skip running prod container"
  else
    echo "[INFO] running prod via runner..."
    "${PROD}/run_prod.sh" run --rm photo_insight
  fi
fi

echo "[OK] deployed to ${PROD} with image tag ${TAG}"