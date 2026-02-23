# Dockerfile
# syntax=docker/dockerfile:1

# ============================================================
# base: runtime deps (requirements.txt) + OS libs
# ============================================================
FROM python:3.10-slim AS base

WORKDIR /app

# ---- Environment sanity ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- OS deps for OpenCV / image stack ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip tooling ----
RUN python -m pip install --upgrade pip setuptools wheel

# ---- Install runtime deps first (cache-friendly) ----
COPY requirements.txt /app/requirements.txt
RUN pip install --prefer-binary -r /app/requirements.txt

# ---- Copy source (runtime stage can run tools too if needed) ----
COPY . /app

CMD ["bash"]


# ============================================================
# ci: base + dev deps (requirements-dev.txt)
#   - for GitHub Actions / running pytest / ruff
# ============================================================
FROM base AS ci

COPY requirements-dev.txt /app/requirements-dev.txt
RUN pip install --prefer-binary -r /app/requirements-dev.txt

CMD ["bash"]


# ============================================================
# dev: ci + developer conveniences (optional)
#   - for VS Code Dev Containers
# ============================================================
FROM ci AS dev

# "git" and "make" are helpful in devcontainers; add here to avoid bloating CI too much.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]