FROM python:3.11-slim

WORKDIR /app

# OpenCV(cv2) が必要とする OS ライブラリを入れる
# libxcb1 が今回のエラーの本命
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# pip を強化
RUN python -m pip install --upgrade pip setuptools wheel

# 依存を先に入れてキャッシュ効かせる
COPY requirements.txt /app/requirements.txt
COPY requirements-dev.txt /app/requirements-dev.txt

RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir -r /app/requirements-dev.txt

# ソース
COPY . /app

CMD ["bash"]
