# Runtime image for the LangGraph RAG QA and chat services.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip \
    && /opt/venv/bin/python -m pip install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_WORKER_COUNT=1 \
    RAG_REPLICA_COUNT=1 \
    WEB_CONCURRENCY=1 \
    CHROMA_DIR=/app/.chroma

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY requirements.txt .
COPY config/ config/
COPY src/ src/

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/.chroma /app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8001

CMD ["python", "-m", "src.qa.main", "serve", "--host", "0.0.0.0"]
