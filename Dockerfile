# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.10

# -----------------------------
# base: OS runtime libs + venv
# -----------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# venv は固定パスで運用（PATH 展開事故を避ける）
ENV VENV_PATH=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Runtime libs: rawpy / opencv(headless) 安定化
RUN apt-get update && apt-get install -y --no-install-recommends \
      libraw20 \
      libglib2.0-0 \
      tini \
    && rm -rf /var/lib/apt/lists/*

# venv 作成（以降は /opt/venv/bin/python, /opt/venv/bin/pip を使う）
RUN python -m venv /opt/venv

WORKDIR /work

# -----------------------------
# builder: build deps + wheels
# -----------------------------
FROM base AS builder

# rawpy がソースビルドに落ちても通るように build deps を入れる
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      pkg-config \
      libraw-dev \
    && rm -rf /var/lib/apt/lists/*

# 依存定義を先にコピーしてキャッシュを効かせる
COPY requirements.txt requirements-dev.txt ./

# wheelhouse を作る（CI/ビルド安定化）
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -U pip wheel setuptools \
    && /opt/venv/bin/pip wheel --wheel-dir /wheels -r requirements-dev.txt

# -----------------------------
# dev: requirements-dev まで入れる（ruff/pytest 含む）
# -----------------------------
FROM base AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
      make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements.txt requirements-dev.txt ./

RUN /opt/venv/bin/pip install --no-index --find-links=/wheels -r requirements-dev.txt \
    && rm -rf /wheels

CMD ["bash"]

# -----------------------------
# ci: dev と同等（実行コマンドを make で切替）
# -----------------------------
FROM dev AS ci

# -----------------------------
# runtime: 最小（requirements のみ）
# -----------------------------
FROM base AS runtime

COPY --from=builder /wheels /wheels
COPY requirements.txt ./

RUN /opt/venv/bin/pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

CMD ["python", "-c", "print('runtime image ready')"]