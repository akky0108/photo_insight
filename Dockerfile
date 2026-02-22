FROM python:3.10-slim

WORKDIR /app

# ---- Environment sanity (recommended) ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- OS deps for OpenCV / image stack ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    make \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip tooling ----
RUN python -m pip install --upgrade pip setuptools wheel

# ---- Install deps first to leverage Docker layer cache ----
COPY requirements.txt /app/requirements.txt
COPY requirements-dev.txt /app/requirements-dev.txt

RUN pip install --prefer-binary -r /app/requirements.txt \
 && pip install --prefer-binary -r /app/requirements-dev.txt

# ---- Copy source ----
COPY . /app

CMD ["bash"]