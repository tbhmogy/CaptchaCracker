# 최신 TensorFlow(2.13.1) 환경을 uv로 구성하는 Dockerfile
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --bin-dir /usr/local/bin

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r uv.lock

COPY . .
ENV PYTHONPATH=/app/src

VOLUME ["/app/data", "/app/model_checkpoints", "/app/documents/training_logs"]

CMD ["uv", "run", "python", "-m", "captcha_cracker_v2.train"]
