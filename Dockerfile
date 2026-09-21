# CrossScan RAG API image, deployed on Hugging Face Spaces (Docker SDK, free
# cpu-basic, 16GB RAM). Postgres (Neon) and Neo4j (Aura) are external managed
# services: set DATABASE_URL, NEO4J_* and GEMINI_API_KEY as Space "Repository
# secrets" (same values as the local .env).
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
# torch's plain PyPI wheel bundles full CUDA support on Linux (several extra GB) unless told
# otherwise - install the CPU-only build from PyTorch's own index FIRST, so the requirements.txt
# install below finds the pinned version already satisfied and never touches the CUDA wheel.
# (Windows PyPI wheels are CPU-only by default, which is why this wasn't visible in local dev.)
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY src ./src

# Model weights (bge, CLIP, reranker, NSFW classifier) download from the HF Hub on first use
# inside this same container - free-tier Spaces have no persistent storage, so this repeats on
# every cold start after the 48h sleep. Expect a slower first request than the ~20s measured
# locally (weights aren't pre-cached), not a broken deployment.
WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
