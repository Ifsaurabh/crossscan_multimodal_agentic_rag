# CrossScan Streamlit chat UI, deployed on Cloud Run (continuous deployment
# from GitHub via Cloud Build). Postgres (Neon) and Neo4j (Aura) are external
# managed services: set DATABASE_URL, NEO4J_* and GEMINI_API_KEY as env vars/
# secrets on the Cloud Run service (same values as the local .env).
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
# Source figures the chat UI shows alongside answers (render_extras() in chat_app.py). Only
# this subfolder - .dockerignore excludes the rest of data/ (raw PDFs, embeddings, etc.), which
# aren't needed at runtime. Missing entirely would fail SOFTLY (chat_app.py checks
# IMAGES_DIR.exists() first), but images just wouldn't show - worth the ~30MB to have them.
COPY data/images ./data/images

# Model weights (bge, CLIP, reranker, NSFW classifier) download from the HF Hub on first use
# inside this same container - Cloud Run's filesystem is ephemeral per instance, so this repeats
# on every cold start after scaling to zero. Expect a slower first request than the ~20s measured
# locally (weights aren't pre-cached), not a broken deployment.
WORKDIR /app/src
EXPOSE 8000
# Cloud Run injects its own PORT env var (default 8080) and health-checks THAT port, not a
# fixed one - a hardcoded --port here made the first real deploy fail ("container failed to
# start and listen on the port... PORT=8080") even though the app itself was fine. Shell form
# (not exec-form JSON array) so $PORT actually expands; falls back to 8000 for local/non-Cloud-Run use.
# enableCORS/enableXsrfProtection=false: Streamlit's defaults assume no reverse proxy in front of
# it: Cloud Run terminates TLS and forwards requests such that the Origin header doesn't match
# what Streamlit expects, which breaks its websocket connection (the whole app) unless disabled.
CMD streamlit run chat_app.py --server.port=${PORT:-8000} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
