# CrossScan RAG API image. Not built/tested yet - written for deployment.
# Postgres and Neo4j are external services: set DATABASE_URL, NEO4J_* and
# GEMINI_API_KEY (and optionally LANGFUSE_*) via the environment / --env-file.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src

# Model weights (bge, CLIP, reranker) download on first use; mount a volume
# at /root/.cache/huggingface to persist them across container restarts.
WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
