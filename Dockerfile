# Customer Support Ticket Router — Dockerfile
# Standalone build using python:3.10-slim (no openenv-base dependency)
# Build context: ticket_router/ directory

FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy full environment source
COPY . /app/env

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/env /app/env

# Python path so `from ticket_router import ...` works inside container
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Enable OpenEnv web interface (Gradio UI)
ENV ENABLE_WEB_INTERFACE=true

# Pass-through env vars (override at runtime: -e HF_TOKEN=... -e MODEL_NAME=...)
ENV HF_TOKEN=""
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
