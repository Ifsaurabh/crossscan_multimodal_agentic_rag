# CrossScan Streamlit chat UI, deployed on Cloud Run by GitHub Actions (.github/workflows/tests.yml):
# tests pass -> this image is built and pushed to Artifact Registry -> a no-traffic revision is
# health-checked -> traffic moves to it. Postgres (Neon) and Neo4j (Aura) are external managed
# services: set DATABASE_URL, NEO4J_*, GEMINI_API_KEY and the ADMIN_* / limit settings as env
# vars/secrets on the Cloud Run service (same names as the local .env).
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
# torch's plain PyPI wheel bundles full CUDA support on Linux (several extra GB) unless told
# otherwise - install the CPU-only build from PyTorch's own index FIRST, so the requirements.txt
# install below finds the pinned version already satisfied and never touches the CUDA wheel.
# (Windows PyPI wheels are CPU-only by default, which is why this wasn't visible in local dev.)
# torchvision is installed here too (not left to requirements.txt): it's only a transitive
# dependency (via docling/sentence-transformers) with no version pinned anywhere, so pip was
# resolving it from plain PyPI - a build whose compiled ops don't match the CPU-only torch 2.14.0
# build above ("RuntimeError: operator torchvision::nms does not exist" at runtime). Installing it
# from the same PyTorch CPU index as torch makes pip resolve a version actually paired with it.
RUN pip install torch==2.14.0 torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# Model weights are BAKED INTO THE IMAGE, so a cold start never downloads anything (the first
# question used to wait minutes for ~1.5 GB from the Hugging Face Hub). Only the two models the
# running app loads are baked: the question-embedding model and the reranker. CLIP and the other
# ingestion-only models are not needed at runtime. The names come from the same config files the
# app reads, so they cannot drift. The two small config files are copied BEFORE the rest of src/,
# so this slow layer stays cached unless a model name changes.
ENV HF_HOME=/opt/hf_cache
COPY src/embedding_config.py src/retrieval_config.py ./src/
RUN cd src && python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
from embedding_config import TEXT_MODEL_NAME; \
from retrieval_config import RERANKER_MODEL; \
SentenceTransformer(TEXT_MODEL_NAME); CrossEncoder(RERANKER_MODEL); \
print('baked:', TEXT_MODEL_NAME, RERANKER_MODEL)"
# From here on the Hub is never contacted: a missing model fails fast and loudly instead of a slow
# download. The build itself proves both models load offline.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
RUN cd src && python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
from embedding_config import TEXT_MODEL_NAME; \
from retrieval_config import RERANKER_MODEL; \
SentenceTransformer(TEXT_MODEL_NAME); CrossEncoder(RERANKER_MODEL); \
print('offline load OK')"

COPY src ./src
# Source figures the chat UI shows alongside answers (render_extras() in chat_app.py). Only
# this subfolder - .dockerignore excludes the rest of data/ (raw PDFs, embeddings, etc.), which
# aren't needed at runtime. Missing entirely would fail SOFTLY (chat_app.py checks
# IMAGES_DIR.exists() first), but images just wouldn't show - worth the ~30MB to have them.
COPY data/images ./data/images

WORKDIR /app/src
EXPOSE 8000
# Cloud Run injects its own PORT env var (default 8080) and health-checks THAT port, not a
# fixed one - a hardcoded --port here made the first real deploy fail ("container failed to
# start and listen on the port... PORT=8080") even though the app itself was fine. Shell form
# (not exec-form JSON array) so $PORT actually expands; falls back to 8000 for local/non-Cloud-Run use.
# enableCORS/enableXsrfProtection=false: Streamlit's defaults assume no reverse proxy in front of
# it: Cloud Run terminates TLS and forwards requests such that the Origin header doesn't match
# what Streamlit expects, which breaks its websocket connection (the whole app) unless disabled.
# showErrorDetails=none: an uncaught error shows users a plain message, never a stack trace (which
# can expose host names and internals); the real error is still written to the Cloud Run logs.
CMD streamlit run chat_app.py --server.port=${PORT:-8000} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --client.showErrorDetails=none
