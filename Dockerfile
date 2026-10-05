# CrossScan Streamlit chat UI, deployed on Cloud Run by GitHub Actions (.github/workflows/tests.yml):
# tests pass -> this image is built and pushed to Artifact Registry -> a no-traffic revision is
# health-checked -> traffic moves to it. Postgres (Neon) is an external managed
# service: set DATABASE_URL, GEMINI_API_KEY and the ADMIN_* / limit settings as env
# vars/secrets on the Cloud Run service (same names as the local .env).
#
# This is the APP image: it contains src/shared/ (used by the app and the worker) and src/retrieval/ (the
# chat and the retrieval pipeline). The ingestion worker has its own image; evaluation is in neither.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

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
COPY src/shared/__init__.py src/shared/embedding_config.py src/shared/model_config.py src/shared/query_guardrail.py ./src/shared/
COPY src/retrieval/__init__.py src/retrieval/retrieval_config.py ./src/retrieval/
RUN cd src && python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
from shared.embedding_config import TEXT_MODEL_NAME; \
from retrieval.retrieval_config import RERANKER_MODEL; \
SentenceTransformer(TEXT_MODEL_NAME); CrossEncoder(RERANKER_MODEL); \
print('baked:', TEXT_MODEL_NAME, RERANKER_MODEL)"
# Llama Prompt Guard 2 (the prompt-injection flagger in query_guardrail.py) is a GATED model, so it
# needs a Hugging Face token with the licence accepted. The token arrives as a BuildKit secret
# (docker build --secret id=hf_token,...): it is mounted for this one step and never stored in a layer.
# No token: a deploy build (REQUIRE_PROMPT_GUARD=1) fails loudly, so production cannot silently ship
# without it; any other build (pull requests, Dependabot - neither can read secrets) warns and
# continues, and the app then runs with injection flagging off, as it does whenever the model is missing.
ARG REQUIRE_PROMPT_GUARD=0
RUN --mount=type=secret,id=hf_token \
    if [ -s /run/secrets/hf_token ]; then \
        cd src && HF_TOKEN="$(cat /run/secrets/hf_token)" python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
from shared.query_guardrail import PROMPT_GUARD_MODEL; \
AutoTokenizer.from_pretrained(PROMPT_GUARD_MODEL); AutoModelForSequenceClassification.from_pretrained(PROMPT_GUARD_MODEL); \
print('baked:', PROMPT_GUARD_MODEL)"; \
    elif [ "$REQUIRE_PROMPT_GUARD" = "1" ]; then \
        echo "ERROR: no hf_token build secret, and this build requires Prompt Guard" >&2; exit 1; \
    else \
        echo "WARNING: no hf_token build secret - Prompt Guard NOT baked into this image"; \
    fi
# From here on the Hub is never contacted: a missing model fails fast and loudly instead of a slow
# download. The build itself proves both models load offline.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
RUN cd src && python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
from shared.embedding_config import TEXT_MODEL_NAME; \
from retrieval.retrieval_config import RERANKER_MODEL; \
SentenceTransformer(TEXT_MODEL_NAME); CrossEncoder(RERANKER_MODEL); \
print('offline load OK')"
# Presidio's spaCy model comes from requirements.txt (a wheel, no download at runtime); prove it loads.
RUN python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model OK')"

COPY src/shared ./src/shared
COPY src/retrieval ./src/retrieval
# The figures the chat UI shows next to answers are NOT baked into the image: the ingestion worker writes them to the
# documents bucket and retrieval/image_store.py reads them from there (the service account needs read access to the
# bucket), so a newly ingested document's figures appear without a redeploy.

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
CMD streamlit run retrieval/chat_app.py --server.port=${PORT:-8000} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --client.showErrorDetails=none
