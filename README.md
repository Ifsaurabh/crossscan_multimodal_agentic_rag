# CrossScan

(Project folder name is still `RAG-NEW` pending a rename.)

An agentic, multimodal RAG project over a corpus of 12 research papers (lung cancer / medical imaging, land cover / remote sensing, and 1 unrelated AI-security paper). Full rationale for every decision lives in `plannings.md` — this file is a short, part-wise summary of *what* was built and *why*.

## Part 1: Ingestion Pipeline (complete)

Turns raw PDFs into embedded, tagged, queryable data in pgvector + Neo4j.

| Stage | What | Why this choice |
|---|---|---|
| **1. Data Loading** | `kagglehub` download, no binary stored in repo | Reproducible from code alone; nothing to commit or go stale |
| **2. Extraction** | Docling (text, block-based with real page numbers) + PyMuPDF (images) | Docling handles multi-column academic layout correctly; page numbers needed later for citations and image-section linking |
| **3. Document Preparation** | Split into sections with page ranges, references stripped | References add no retrieval value; section labels enable citation-quality answers and later graph structure |
| **3b. Ingestion Guardrails** | Email PII redaction (text) + NSFW check (images, Falconsai model) | Documents may contain author emails; images should be screened even though risk is low for this corpus |
| **4. Chunking** | Parent-child (custom grouping + RecursiveCharacterTextSplitter fallback) | Small chunks (children) for precise search, larger chunks (parents) returned for context — search precision and answer quality are different problems |
| **5. Embedding** | `bge-base-en-v1.5` (text) + CLIP `ViT-B/32` (images), **structure-aware** (document/section prefixed before embedding) | Two separate models beat one shared multimodal model on text-retrieval quality; structure-awareness fixes ambiguous short chunks that read identically across different papers |
| **6. Vector Storage** | pgvector on a new dedicated local Postgres database | Deployment-ready (this project needs to go live later) without introducing a new service; reuses infra the user already runs |
| **6b. Graph Storage** | Neo4j — Paper/Method/Dataset/Metric/Section/Image nodes | Captures cross-paper relationships (shared methodology across domains) that vector similarity search can't express directly |

**Numbers**: 12 papers → 316 parent chunks, 625 embedded child chunks, 174 embedded + graph-linked images, 12 Paper nodes, ~99 entity relationships, 316 Section nodes. 565 tests (529 unit, 33 integration against a real Postgres, 3 live smoke tests), all passing.

**Token usage & latency** (measured directly, `cl100k_base` via `tiktoken`; latency is wall-clock, CPU-only, no GPU):

| Stage | Tokens | Latency |
|---|---|---|
| Chunking (parents) | 150,472 tokens across 316 parent chunks | 0.9s (no ML — pure tokenization/grouping) |
| Chunking (children, what gets embedded) | 156,344 tokens across 625 child chunks | included above |
| Text embedding (bge-base-en-v1.5) | 156,344 tokens embedded | 703.7s (~11.7 min) |
| Image embedding (CLIP ViT-B/32) | n/a (visual, not token-based) | 45.1s |
| Entity extraction (Gemini API, hosted) | not measured (API token accounting not captured) | 141.2s (~2.4 min) across 11 documents |

Text embedding is the slowest stage by far — expected, since it's a ~110M-parameter transformer model run on CPU across 625 chunks, versus chunking/grouping which is pure Python logic.

### Notable decisions and detours

- **Domain classification (embedding-similarity, not an LLM)**: A local LLM (Qwen2.5 3B) was tried first for tagging each paper's domain but produced inconsistent, sometimes wrong results (a 3B model isn't reliable for this kind of sequential classification). Reusing the already-computed text embeddings with a confidence threshold — falling back to a *new* domain instead of forcing a bad match — correctly classified all 12 documents, including catching that one paper is actually unrelated to either expected domain.

- **Entity extraction (Gemini, not local)**: Qwen was tried again here and confirmed to **hallucinate** — it invented specific version numbers not in the source text, and fabricated an entire methods list for a paper that names none. Switched to Google Gemini (free tier) with a hardened anti-hallucination prompt and a verification step (checks each extracted entity actually appears in the source text). Both previously-bad documents are now handled correctly, including one that correctly returns *zero* entities rather than inventing content. This is the project's one deliberate exception to an otherwise free/local-only model policy — justified by a real, confirmed quality problem, not preference.

### Known library bug worked around

- **`CLIPModel.get_image_features()` returns the wrong object in `transformers==5.17.0`** — a raw intermediate vision-encoder output instead of the documented 512-dim projected embedding, with no attribute to recover the correct value. Silently corrupted all image embeddings until the database insert failed. Fixed in `src/embed_images.py` by calling `model.vision_model(...)` + `model.visual_projection(...)` directly (what `get_image_features()` is supposed to do internally). Re-verify if upgrading `transformers`.

## Part 2: Retrieval Pipeline (complete, live-verified)

Answers live questions against the ingested data. Wired as a LangGraph `StateGraph` with two LLM agents and deterministic (non-LLM) guardrail tools.

| Stage | What | Why this choice |
|---|---|---|
| **7. Input Guardrail** | Regex PII redaction, prompt-injection and medical-advice-framing detection (`query_guardrail.py`) | Guardrails are deterministic tools, not agents: sending a query to an LLM just to check whether it is safe to send to an LLM is self-defeating (and would leak the PII being redacted) |
| **Query cache** | Two-tier Postgres cache: raw query, then transformed query (`query_cache.py`) | Point lookups belong in Postgres; a graph DB is built for traversal and would be slower than the work it saves. Doubles as a query log |
| **8a. Agent 1: Transform + Route** | Decomposes compound questions, generates up to 3 phrasing variants, and routes each sub-query: no retrieval / vector / graph / both, simple or hybrid search, images needed or not | Mixed questions ("which papers use X and what is its accuracy") need different routing per part |
| **8b. Retrieval executor** | pgvector semantic search + Postgres full-text (RRF fusion) for hybrid, Neo4j entity lookup for graph, image lookup via graph relationships | Hybrid keyword + semantic recovers exact acronyms embeddings blur. Graph "hybrid" was rejected: entities are exact-name matches, there is no semantic signal to fuse |
| **8c. Reranker** | `BAAI/bge-reranker-base` cross-encoder (`reranker.py`) | Local and free, same model family as the embedder |
| **9. Agent 2: Quality check + Generate** | Judges whether retrieved context suffices, then writes an answer with inline `[paper.pdf, p.N]` citations | Judging before generating lets the system retry instead of answering from bad context |
| **Retry policy** | Max 3 attempts: normal, then rerank, then re-route with structured feedback ("what was missing / what to look for"), then answer with a low-confidence caveat | Bounded loop, no infinite-retry risk; the retry is informed, not blind |
| **General-knowledge path** | Sub-queries that need no retrieval skip the retry loop and are answered with an explicit "not from the knowledge base" label | Users must always be able to tell corpus-grounded answers from model-knowledge answers |
| **10. Output Guardrail** | Verifies every citation against what was actually retrieved, flags system-prompt leaks and clinical overstatement (`output_guardrail.py`) | Reuses the "verify against source" pattern from entity extraction; deterministic, no LLM |
| **LLM connection** | One module, `src/llm_connection.py`, is the only place the app talks to a model, and only API keys come from `.env`. Models are grouped into **three tiers of two models each** (a primary and a fallback): **fast** (planner, quality check, summaries, simple answers), **reasoning** (complex answers) and **evaluation** (DeepEval / RAGAS judges). A task just names its tier; the connection tries the tier's primary, then its fallback, and a model that fails (quota, outage) goes on a short per-model cooldown. A tier never borrows another tier's models. All tiers use Google models for now (`gemini-3.6-flash` / `3.5-flash-lite`, `3.8-flash` / `3.7-flash`, `3.7-flash` / `3.5-flash`); switching a tier to another vendor is a settings edit | Cheap models for mechanical steps and stronger ones only where reasoning matters keeps cost down; a separate judge tier keeps evaluation independent; fallback within a tier keeps the app answering when one model's free quota runs out (quotas are per model). The query planner labels each sub-question simple or complex at no extra model call, and the answer writer uses the fast or reasoning tier to match. Ingestion's entity extraction still calls Gemini directly |
| **Resilience** | Retry-with-backoff on Gemini 503s (`gemini_retry.py`), logging the real status code; a model with a healthy fallback in its tier gets one retry and then fails over (the last model keeps 4 attempts); prompt caching is attempted only for instructions of 1,024+ tokens (Gemini refuses less) | Driven by real failures seen in live runs. A 503 is Google-side overload, not a quota error (429), so the two are classified and cooled down differently |

**Verified live** (real Postgres + Neo4j + Gemini): vector-type, graph+vector-type and general-knowledge queries all returned correct, cited, correctly labelled answers. Live runs also caught two bugs unit tests could not (multi-page citation format; transient 503s).

## Part 3: Evaluation and MLOps

| Piece | What | Why this choice |
|---|---|---|
| **Golden set** | `data/golden_set.jsonl`: one grounded Q&A + source context per paper (36 examples, 3 per paper across 3 domains: v1's 12, plus 24 more hand-written from other sections) | Hand-authored for v1 because DeepEval's Synthesizer needs many LLM calls per example and hit Gemini's free-tier daily quota. The Synthesizer code (`generate_golden_set.py`, resumable) stays for scaling the set up later |
| **Dataset versioning: DVC** | Golden set tracked by DVC, pushed to a local remote; git holds only the `.dvc` pointer | Industry-standard data versioning alongside git. The DVC md5 is the golden-set version recorded in every eval run |
| **Prompt management: Langfuse** | Agent instructions fetched from Langfuse Prompt Management (`production` label) with the local constant as fallback; `python src/prompt_registry.py` pushes local changes as new versions | Versioned, auditable prompts without a redeploy. Fallback means observability can never break the pipeline |
| **Tracing: Langfuse** | Self-hosted (Docker), LangGraph runs traced through a callback handler | Free and local, matching the project's other infrastructure |
| **Experiment tracking: MLflow** | `run_evaluation.py` runs the golden set and logs one MLflow run: golden-set md5, prompt hashes, retrieval config, metrics, per-row results | A score is only meaningful if you know exactly which data, prompts and config produced it |
| **Metrics** | Always-on, no LLM: source hit, citation-verified rate, grounded rate, guardrail flags, latency, token usage. Optional LLM-judge (`--llm-metrics`): DeepEval (faithfulness, answer relevancy, contextual precision/recall) and RAGAS (faithfulness, context precision/recall) | Deterministic metrics scale to any corpus size for free; LLM judges cost quota, so they are opt-in |
| **Online evaluation** | Thumbs up/down under every answer; about 10% of live answers (`ONLINE_EVAL_SAMPLE_RATE`) are scored for faithfulness and relevance by an evaluation-tier LLM judge in a background thread; `python src/online_report.py` (also an admin panel and `GET /admin/online-metrics`) gives latency, cache/block rates, satisfaction, judge scores, alerts, and a review queue of bad answers | Offline evaluation says a version is safe to ship; online evaluation says it still works on questions nobody wrote. The review queue feeds new cases back into the golden set. The judge never delays or charges the user |
| **Cost tracking** | Gemini token usage recorded per call (`usage_tracker.py`); cost uses `GEMINI_PRICE_PER_M_INPUT/OUTPUT` (0 on the free tier) | Tokens are measured; dollars are only as accurate as the prices you configure |

## Part 4: Chatbot, Memory and Access Control

| Piece | What | Why this choice |
|---|---|---|
| **Chat UI** | Streamlit (`src/chat_app.py`): sign-in, chat with sources/images/guardrail notes, chat history sidebar, quota meter, saved notes | The most widely used framework for RAG demos; talks to the same service layer as the API, in one process |
| **Authentication** | Username + password, hashed with scrypt (standard library, per-user salt). Login issues a random token; only its SHA-256 is stored, with an expiry; logout and password reset revoke it. Generic failure messages, constant-time checks, per-username lockout after repeated failures | No extra crypto dependency; a leaked database yields neither passwords nor usable tokens; server-side sessions can be revoked instantly (unlike stateless JWTs) |
| **Authorization** | Two roles, `user` and `admin`. Every chat session, message and note query is scoped by `user_id`; another user's session id returns 404, indistinguishable from a missing one. Admin-only: user management, `/stats`. Self-registration is on by default (open sign-up; `ALLOW_REGISTRATION=0` turns it off), throttled per hour, and never grants admin | Conversations cannot mix between users, and ownership is enforced in SQL, not only in the UI |
| **Quotas and limits** | Per-user daily messages and tokens (regular users get 2 messages/day for now: a first question and one follow-up; role defaults, per-user overrides), a per-minute rate limit, a mandatory global daily cap across all users (default 50), at most 20 different users per day (`DAILY_ACTIVE_USER_CAP`; anyone already served today is never cut off), and at most 3 questions processed at once (`MAX_CONCURRENT_REQUESTS`; a full house answers "busy, retry in a few seconds" without charging) | Protects a small upstream quota (Gemini free tier: about 20 calls/day per model; one message is 3-5 calls). Daily limits are checked before the rate limiter so refused requests never use a slot. Only real answers are charged: a failed run, an outage, or a message blocked before any model call costs the user nothing |
| **Conversation memory** | Sessions and messages in Postgres. The last 6 messages go to Agent 1, which resolves follow-ups ("what about its accuracy?") into self-contained sub-queries | Folding this into the existing Agent 1 call costs no extra LLM call. Follow-ups and any request with user notes bypass the shared query cache, so a context-dependent question can never be served (or stored) as an answer to the bare text |
| **Rolling summary** | When 6 or more messages age out of the window, one LLM call folds them into a running summary | Keeps context bounded on long chats; best-effort, so a failure or exhausted quota never fails the turn |
| **Long-term memory** | `/remember <text>`, `/memories`, `/forget <n\|all>`. Notes are embedded (pgvector) and the relevant ones are recalled per question | Explicit consent only: nothing is remembered automatically. Notes are PII-redacted, capped (50 per user), refused if they look like prompt injection, and shown to the model only as background about the user, never as facts about the papers |
| **API** | FastAPI (`src/api.py`): `/auth/*`, `/me`, `/chat`, `/chat/sessions*`, `/memories*`, `/admin/*`, `/stats`, `/health`. Run with `uvicorn api:app` from `src/` | Same service layer as the UI, for programmatic use |
| **Admin CLI** | `python src/manage_users.py` (`setup-db`, `create-user`, `list-users`, `set-limits`, `deactivate`, `activate`, `reset-password`) | Passwords are prompted, not passed on the command line |
| **Deployment** | `Dockerfile` for the API (Postgres, Neo4j and Gemini are external, configured by environment variables) | |

**First-time setup**: `python src/setup_app_db.py`, then `python src/manage_users.py create-user <name> --role admin`, then `streamlit run src/chat_app.py`. All quota/limit settings are in `.env.example`.

Run the evaluation: `python src/run_evaluation.py [--limit N] [--llm-metrics none|deepeval|ragas|both]`. View runs: `mlflow ui --backend-store-uri ./mlruns`.

**Tests**: `tests/unit/` (fully faked; a guard fails any test that reaches real Postgres, Neo4j or Gemini), `tests/integration/` (real Postgres in a throwaway schema; `pytest tests/integration --run-integration`), `tests/live/` (real Gemini and Neo4j, spends quota; `pytest tests/live --run-live`). Plain `pytest tests` runs the unit tests only.

**Status honesty** (2026-09-20): the query pipeline, chat handler (follow-ups, memory, quotas) and model fallback have run end to end against real Postgres, Neo4j and Gemini; the application SQL passes 33 integration tests; the API and the Streamlit server start. **Not yet proven**: a real login and chat over HTTP, the chat UI in a browser (including the feedback buttons), the online judge against a real model, and a full offline evaluation of the golden set (it exceeds one free-tier day). Langfuse tracing is off (its Docker stack was removed; `LANGFUSE_DISABLED=1`), MLflow has never logged a real run, and the Dockerfile has not been built. Two papers have no method/dataset/metric links in the graph (entity extraction produced nothing usable), so graph-style questions can miss them. Not built: password-strength rules beyond length, email verification/reset flows, per-IP rate limiting (limits are per user), cross-process concurrency limiting (it is per process), and image display depends on the extracted image files being found under `data/images/`.

Interview-ready write-ups of the improvements, with measured before/after numbers: `improvement_stories.md`.
