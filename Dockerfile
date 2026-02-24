# syntax=docker/dockerfile:1

FROM python:3.10-slim AS base

WORKDIR /workspaces/photo_insight

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libraw-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt ./requirements.txt
RUN python -m pip install --prefer-binary -r ./requirements.txt


FROM base AS ci
COPY requirements-dev.txt ./requirements-dev.txt
RUN python -m pip install --prefer-binary -r ./requirements-dev.txt

# ★CIで make を回すためにリポジトリを含める
COPY . .


FROM ci AS dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    git make \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]