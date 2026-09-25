---
title: CrossScan
emoji: 🔬
colorFrom: teal
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# CrossScan
An agentic, multimodal RAG project over a corpus of 12 research papers (lung cancer / medical imaging, land cover / remote sensing, and 1 unrelated AI-security paper). This file is a short, part-wise summary of *what* was built and *why*, with measured numbers where they exist.

**Live demo**: https://crossscan-multimodal-agentic-rag-git-807612796446.europe-west1.run.app (sign-in required; open sign-up, 3 messages per day per account).

## What it looks like (live deployment)

| | |
|---|---|
| ![Sign-in page](screenshots/streamlit_homepage.png)<br>**Sign-in**: open sign-up, or accounts created by an administrator | ![Answer with citations](screenshots/question_from_knowledge.png)<br>**A cited answer**: inline `[paper.pdf, p.N]` citations, a guardrail note, and the per-user quota meter (the admin account is unlimited) |
| ![Medical question](screenshots/question_out_of_knowledge.png)<br>**A personal medical question**: answered, but with a fixed "not medical advice" note first and a label that the answer is not from the knowledge base | ![Prompt injection blocked](screenshots/prompt_injection_response.png)<br>**Prompt injection blocked** before any model call, plus the admin "live quality" panel (latency, cache and block rates) |
| ![Sources and cache](screenshots/cache_hit_test_question.png)<br>**Sources panel**, and a repeated question served from the answer cache | ![Latency report](screenshots/latency%20for%20last%20few%20questions.png)<br>**Per-question latency**, exported from Langfuse traces |

## Part 1: Ingestion Pipeline (complete)

Turns raw PDFs into embedded, tagged, queryable data in pgvector + Neo4j.

| Stage | What | Why this choice |
|---|---|---|
| **1. Data Loading** | `kagglehub` download, no binary stored in repo | Reproducible from code alone; nothing to commit or go stale |
| **2. Extraction** | Docling (text and **tables**, block-based with real page numbers) + PyMuPDF (images) | Docling handles multi-column academic layout correctly; page numbers needed later for citations and image-section linking |
| **3. Document Preparation** | Split into sections with page ranges, tables set aside (they become graph nodes, not prose), references stripped | References add no retrieval value; section labels enable citation-quality answers and later graph structure |
| **Entity extraction** | Gemini extracts each paper's methods, datasets, metrics and baselines; every entity is verified against the source text before it is kept | A local model hallucinated here (see below); verification removes anything the paper does not literally contain |
| **Domain classification** | One batched Gemini call (fast tier) reads the first ~800 characters of every paper and returns its title and a 2-4 word research domain | One call for up to 100 papers; no per-domain seed labels to maintain when the corpus changes |
| **3b. Ingestion Guardrails** | Email PII redaction (text) + injection-pattern flagging (text, domain-aware) | Documents may contain author emails; a paper about AI security may legitimately discuss injection patterns, so those get flagged with domain context for review, not blocked |
| **4. Chunking** | Parent-child with LangChain's `RecursiveCharacterTextSplitter` counting real tokens (`tiktoken`): parents about 1,800 tokens, children about 400, 20% overlap | Small chunks (children) for precise search, larger chunks (parents) returned for context — search precision and answer quality are different problems |
| **5. Embedding** | `bge-base-en-v1.5` (text) + CLIP `ViT-B/32` (images), **structure-aware** (document/section prefixed before embedding) | Two separate models beat one shared multimodal model on text-retrieval quality; structure-awareness fixes ambiguous short chunks that read identically across different papers |
| **6. Vector Storage** | pgvector on Postgres: a local database for development, Neon (managed) in production | One database for vectors, full-text search, chat history and quotas; a separate vector store was not needed at this scale (the 12 papers use about 17 MB of a 500 MB free tier) |
| **6b. Graph Storage** | Neo4j: Paper, Method, Dataset, Metric, Baseline, Section, Image and **Table** nodes (a table's text is stored on its node and linked to the section it sits in by page range) | Captures cross-paper relationships (shared methodology across domains) that vector similarity search can't express directly |
| **Orchestration** | `run_ingestion.py` runs all 12 stages in order, records every run and stage (status, time, tokens) in Postgres, stops at the first failure and resumes from it next time | Unattended re-runs must not redo hours of work. A content-hash manifest for detecting changed documents (`ingestion_manifest.py`, `check_new_documents.py`) is written and tested but deliberately not wired in yet |

**Numbers**: 12 papers → 316 parent chunks, 630 embedded child chunks, 182 embedded and graph-linked images, 53 tables (as graph nodes), 12 Paper nodes, 316 Section nodes, 88 entity nodes (65 methods, 7 datasets, 16 metrics). 923 unit tests, all passing, plus integration tests against a real Postgres (run in CI) and opt-in live smoke tests.

**Token usage & latency** (measured at the first full ingestion run, `cl100k_base` via `tiktoken`; latency is wall-clock, CPU-only, no GPU):

| Stage | Tokens | Latency |
|---|---|---|
| Chunking (parents) | 150,472 tokens across 316 parent chunks | 0.9s (no ML — pure tokenization/grouping) |
| Chunking (children, what gets embedded) | 156,344 tokens across 625 child chunks | included above |
| Text embedding (bge-base-en-v1.5) | 156,344 tokens embedded | 703.7s (~11.7 min) |
| Image embedding (CLIP ViT-B/32) | n/a (visual, not token-based) | 45.1s |
| Entity extraction (Gemini API, hosted) | not measured (API token accounting not captured) | 141.2s (~2.4 min) across 11 documents |

Text embedding is the slowest stage by far — expected, since it's a ~110M-parameter transformer model run on CPU across 625 chunks, versus chunking/grouping which is pure Python logic.

### Notable decisions and detours

- **Domain classification (one batched LLM call)**: Two earlier designs were dropped. A local 3B model (Qwen2.5) gave inconsistent, sometimes wrong tags. An embedding-similarity version worked for the 12 papers but relied on hand-written seed labels per domain, which does not carry over to a new corpus. It is now a single batched Gemini call over each paper's opening text, returning a title and a domain; it correctly separates the unrelated AI-security paper from the two expected domains.

- **Entity extraction (Gemini, not local)**: Qwen was tried again here and confirmed to **hallucinate** — it invented specific version numbers not in the source text, and fabricated an entire methods list for a paper that names none. Switched to Google Gemini (free tier) with a hardened anti-hallucination prompt and a verification step (checks each extracted entity actually appears in the source text). Both previously-bad documents are now handled correctly, including one that correctly returns *zero* entities rather than inventing content. It also extracts **baselines** (methods a paper compares against) and keeps a paper's own method out of that list. This is the project's one deliberate exception to an otherwise free/local-only model policy — justified by a real, confirmed quality problem, not preference.

### Known library bug worked around

- **`CLIPModel.get_image_features()` returns the wrong object in `transformers==5.17.0`** — a raw intermediate vision-encoder output instead of the documented 512-dim projected embedding, with no attribute to recover the correct value. Silently corrupted all image embeddings until the database insert failed. Fixed in `src/embed_images.py` by calling `model.vision_model(...)` + `model.visual_projection(...)` directly (what `get_image_features()` is supposed to do internally). Re-verify if upgrading `transformers`.

## Part 2: Retrieval Pipeline (complete, live-verified)

Answers live questions against the ingested data. Wired as a LangGraph `StateGraph` with two LLM agents and deterministic (non-LLM) guardrail tools.

| Stage | What | Why this choice |
|---|---|---|
| **7. Input Guardrail** | Regex PII redaction, prompt-injection and medical-advice-framing detection (`query_guardrail.py`) | Guardrails are deterministic tools, not agents: sending a query to an LLM just to check whether it is safe to send to an LLM is self-defeating (and would leak the PII being redacted) |
| **Query cache** | Two-tier Postgres cache: raw query, then transformed query (`query_cache.py`) | Point lookups belong in Postgres; a graph DB is built for traversal and would be slower than the work it saves. Doubles as a query log |
| **8a. Agent 1: Transform + Route** | Decomposes compound questions, generates up to 3 phrasing variants, and routes each sub-query: no retrieval / vector / graph / both, simple or hybrid search, images needed or not | Mixed questions ("which papers use X and what is its accuracy") need different routing per part |
| **8b. Retrieval executor** | pgvector semantic search + Postgres full-text (RRF fusion) for hybrid, Neo4j entity lookup for graph (the graph acts as a **filter**: it names the papers and a vector search runs inside them, falling back to the normal search when it finds nothing), and image and **table** lookup via graph relationships | Hybrid keyword + semantic recovers exact acronyms embeddings blur. Graph "hybrid" was rejected: entities are exact-name matches, there is no semantic signal to fuse |
| **8c. Reranker** | `BAAI/bge-reranker-base` cross-encoder (`reranker.py`) | Local and free, same model family as the embedder |
| **9. Agent 2: Quality check + Generate** | Judges whether retrieved context suffices, then writes an answer with inline `[paper.pdf, p.N]` citations | Judging before generating lets the system retry instead of answering from bad context |
| **Retry policy** | Max 3 attempts: normal, then rerank, then re-route with structured feedback ("what was missing / what to look for"), then answer with a low-confidence caveat | Bounded loop, no infinite-retry risk; the retry is informed, not blind |
| **General-knowledge path** | Sub-queries that need no retrieval skip the retry loop and are answered with an explicit "not from the knowledge base" label | Users must always be able to tell corpus-grounded answers from model-knowledge answers |
| **10. Output Guardrail** | Verifies every citation against what was actually retrieved, flags system-prompt leaks and clinical overstatement, and **redacts** emails and credential-shaped strings from the answer (`output_guardrail.py`) | Reuses the "verify against source" pattern from entity extraction; deterministic, no LLM |
| **LLM connection** | One module, `src/llm_connection.py`, is the only place the app talks to a model, and only API keys come from `.env`. Models are grouped into **three tiers of two models each** (a primary and a fallback): **fast** (planner, quality check, summaries, simple answers), **reasoning** (complex answers) and **evaluation** (DeepEval / RAGAS judges). A task just names its tier; the connection tries the tier's primary, then its fallback, and a model that fails (quota, outage) goes on a short per-model cooldown. A tier never borrows another tier's models. All tiers use Google models for now (`gemini-3.6-flash` / `3.5-flash-lite`, `3.8-flash` / `3.7-flash`, `3.7-flash` / `3.5-flash`); switching a tier to another vendor is a settings edit | Cheap models for mechanical steps and stronger ones only where reasoning matters keeps cost down; a separate judge tier keeps evaluation independent; fallback within a tier keeps the app answering when one model's free quota runs out (quotas are per model). The query planner labels each sub-question simple or complex at no extra model call, and the answer writer uses the fast or reasoning tier to match. Ingestion's entity extraction still calls Gemini directly |
| **Resilience** | Retry-with-backoff on Gemini 503s (`gemini_retry.py`), logging the real status code; a model with a healthy fallback in its tier gets one retry and then fails over (the last model keeps 4 attempts); prompt caching is attempted only for instructions of 1,024+ tokens (Gemini refuses less); every Gemini call has a 30-second timeout so a hung call fails over to the tier's next model; Postgres connections come from a health-checked pool (so Neon's idle suspend cannot hand out a dead connection) and Neo4j uses one shared driver; a database outage shows a friendly "temporarily unavailable, not counted against your limit" message instead of a crash | Driven by real failures seen in live runs. A 503 is Google-side overload, not a quota error (429), so the two are classified and cooled down differently |

**Verified live** (real Postgres + Neo4j + Gemini): vector-type, graph+vector-type and general-knowledge queries all returned correct, cited, correctly labelled answers. Live runs also caught two bugs unit tests could not (multi-page citation format; transient 503s). The deployed app has since been checked end to end on Cloud Run.

### Latency and cost optimization (measured)

Two rounds, each starting from a measurement rather than a guess.

**Round 1: do less work per question (19 Sept, one question instrumented per stage and per model call).** The first live question was correct but slow and expensive. The per-call table showed why: the planner split it into two sub-questions that both needed the *same* results passage, so the same ~7,800-token context went to a model four times (two quality checks and two answers), on the stronger tier.

| | Before | After | Change |
|---|---|---|---|
| Pipeline time | 86.9 s | **58.2 s** | **-33%** |
| Model calls | 5 | **3** | -40% |
| Prompt tokens | 32,015 | **14,609** | **-54%** |
| Output tokens | 1,116 | 581 | -48% |
| Sub-questions | 2 (overlapping) | 1 | |

Fixes, layered so one failing still leaves the others: (1) the planner prompt now splits only when the parts need *different* lookups; (2) a deterministic safety net merges sub-questions whose retrieved passages overlap by 80% or more (measured against the smaller set), because prompts are probabilistic; (3) prompt-cache requests are skipped for instructions under 1,024 tokens, which Gemini refuses anyway; (4) a model with a healthy fallback gets one retry and then fails over, which removes up to about 14 seconds of waiting per busy call. A bug found on the way: the second answer cache is read before retrieval and written after it, so merging sub-questions changed the cache key and merged questions could never hit; the pre-merge key is now kept in the graph state (covered by a test).

*Checked against the raw run reports:* the numbers above match them exactly. The stage tables show where the gain was: quality check plus answer fell from 54.7 s to 8.7 s (-84%), and time inside model calls fell from 66.0 s to 38.6 s, while retrieval (about 20 s) did not change. The planner call itself went from 11.3 s to 29.9 s on that run (Gemini variance), which hides part of the gain, so -33% understates the fix. Limits: one run, and LLM latency varies; the merge safety net has unit tests but was not exercised against a live model. A cold / warm / cached test in the same session found the answer cache answers an exact repeat in 0.17 s with zero model calls, while a warm run was *not* faster (retrieval fell from 17.0 s to 5.2 s as models stayed loaded, but a Gemini 503 and a different plan cost more).

**Round 2: remove the waiting (25-26 Sept, on the way to production).** Langfuse traces of a slow run showed most of the time was not thinking at all: every graph node opened its own database connection (about 10 s each from a laptop, 10-20 per question), the 1.1 GB reranker was downloaded on first use (6 min 55 s), and one Gemini call hung for 70 seconds with no timeout.

| Change | What it does |
|---|---|
| Health-checked Postgres connection pool, one shared Neo4j driver | Connections are opened once and reused; a connection Neon closed while idle is replaced, not lent out |
| Both runtime models baked into the Docker image, Hub access switched off | A cold start downloads nothing |
| 30-second timeout on every Gemini call | A hung call fails over to the tier's next model instead of blocking a minute |
| Cloud Run in europe-west1, with a friendly "temporarily unavailable" message for outages | Shorter round trips to the databases; no stack traces for users |

| Measured | Before (laptop, 25 Sept 17:51 UTC) | After (Cloud Run, 25 Sept 19:10-19:35 UTC) |
|---|---|---|
| A question that misses the cache, end to end | **2 min 27 s** (11 min on the very first run, including the download) | **8-40 s** across 5 full questions, median about 17 s |
| Answer cache check (a database lookup) | 13.5 s | 0.0-0.4 s |
| Retrieval | 47-67 s | 7.7-13.2 s |
| Repeated question answered from the cache | n/a | 1.8 s |

The two columns come from different places (a laptop about 0.5 s from the database, versus Cloud Run about 0.1 s away), so the gain combines pooling, baked-in models and location, and the share of each is not separately measured. The pool alone was measured from the laptop: the first connection took 9.7 s, and every later borrow 1.5-1.9 s. **What is still slow:** retrieval (8-13 s, mostly embedding on 2 CPUs plus several database round trips to a Neon instance in Ohio), and `gemini-3.6-flash`, which failed or took up to 15 s in 2 of 3 traced questions before the tier fell back to `flash-lite` (about 1 s for the same call).

## Part 3: Evaluation and MLOps

| Piece | What | Why this choice |
|---|---|---|
| **Golden set** | `data/golden_set.jsonl`: one grounded Q&A + source context per paper (36 examples, 3 per paper across 3 domains: v1's 12, plus 24 more hand-written from other sections) | Hand-authored for v1 because DeepEval's Synthesizer needs many LLM calls per example and hit Gemini's free-tier daily quota. The Synthesizer code (`generate_golden_set.py`, resumable) stays for scaling the set up later |
| **Dataset versioning: DVC** | Golden set tracked by DVC, pushed to a local remote; git holds only the `.dvc` pointer | Industry-standard data versioning alongside git. The DVC md5 is the golden-set version recorded in every eval run |
| **Prompt management: Langfuse** | Agent instructions fetched from Langfuse Prompt Management (`production` label) with the local constant as fallback; `python src/prompt_registry.py` pushes local changes as new versions | Versioned, auditable prompts without a redeploy. Fallback means observability can never break the pipeline |
| **Tracing: Langfuse** | Langfuse Cloud (free tier), LangGraph runs traced through a callback handler; keys wired in as secrets on both local `.env` and the deployed Cloud Run service | Self-hosting (Docker) was tried first but its stack got wiped and isn't reachable from Cloud Run anyway; Langfuse Cloud works identically for local dev and the live deployment with no infra to run |
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
| **Quotas and limits** | Per-user daily messages and tokens (regular users get 3 messages/day for now; per-user overrides; the admin account from `.env` has no limits at all), a per-minute rate limit, a mandatory global daily cap across all users (default 50), at most 20 different users per day (`DAILY_ACTIVE_USER_CAP`; anyone already served today is never cut off), and at most 3 questions processed at once (`MAX_CONCURRENT_REQUESTS`; a full house answers "busy, retry in a few seconds" without charging) | Protects a small upstream quota (Gemini free tier: about 20 calls/day per model; one message is 3-5 calls). Daily limits are checked before the rate limiter so refused requests never use a slot. Only real answers are charged: a failed run, an outage, or a message blocked before any model call costs the user nothing |
| **Conversation memory** | Sessions and messages in Postgres. The last 6 messages go to Agent 1, which resolves follow-ups ("what about its accuracy?") into self-contained sub-queries | Folding this into the existing Agent 1 call costs no extra LLM call. Follow-ups and any request with user notes bypass the shared query cache, so a context-dependent question can never be served (or stored) as an answer to the bare text |
| **Rolling summary** | When 6 or more messages age out of the window, one LLM call folds them into a running summary | Keeps context bounded on long chats; best-effort, so a failure or exhausted quota never fails the turn |
| **Long-term memory** | `/remember <text>`, `/memories`, `/forget <n\|all>`. Notes are embedded (pgvector) and the relevant ones are recalled per question | Explicit consent only: nothing is remembered automatically. Notes are PII-redacted, capped (50 per user), refused if they look like prompt injection, and shown to the model only as background about the user, never as facts about the papers |
| **API** | FastAPI (`src/api.py`): `/auth/*`, `/me`, `/chat`, `/chat/sessions*`, `/memories*`, `/admin/*`, `/stats`, `/health`. Run with `uvicorn api:app` from `src/` | Same service layer as the UI, for programmatic use |
| **Admin CLI** | `python src/manage_users.py` (`setup-db`, `create-user`, `list-users`, `set-limits`, `deactivate`, `activate`, `reset-password`) | Passwords are prompted, not passed on the command line |
| **Deployment** | `Dockerfile` runs the Streamlit chat UI (`chat_app.py`), deployed on Google Cloud Run by GitHub Actions: a push to `main` runs the tests and, only if they pass, builds the image (the two runtime models are baked in, so a cold start downloads nothing), deploys it as a no-traffic revision, health-checks it and then moves traffic to it (`.github/workflows/tests.yml`; one-time Google Cloud setup in `.github/DEPLOY_SETUP.md`). Pull requests also build and start the image (`docker-build.yml`); a nightly full-suite run and a weekly live-evaluation run are separate workflows. Postgres (Neon), Neo4j (Aura Free), Gemini and Langfuse are all external, configured via Secret Manager | Live demo: https://crossscan-multimodal-agentic-rag-git-807612796446.europe-west1.run.app |

**First-time setup**: `python src/setup_app_db.py`, then put `ADMIN_USERNAME` / `ADMIN_PASSWORD` in `.env` (the admin account is created and kept in sync at startup; `manage_users.py create-user <name> --role admin` still works for extra admins), then `streamlit run src/chat_app.py`. All quota/limit settings are in `.env.example`.

Run the evaluation: `python src/run_evaluation.py [--limit N] [--llm-metrics none|deepeval|ragas|both]`. View runs: `mlflow ui --backend-store-uri ./mlruns`.

**Tests**: `tests/unit/` (fully faked; a guard fails any test that reaches real Postgres, Neo4j or Gemini), `tests/integration/` (real Postgres in a throwaway schema; `pytest tests/integration --run-integration`), `tests/live/` (real Gemini and Neo4j, spends quota; `pytest tests/live --run-live`). Plain `pytest tests` runs the unit tests only (923 today).

**Status honesty** (2026-09-26): the query pipeline, chat handler (follow-ups, memory, quotas) and model fallback have run end to end against real Postgres, Neo4j and Gemini, and the app is deployed and answering real questions over HTTPS on Cloud Run through a tested, health-gated GitHub Actions pipeline, with Langfuse Cloud tracing and prompt management live. **Measured live** (Langfuse, 2026-09-25): a full question takes about 8-40 seconds (it took 2.5-11 minutes before pooled connections, baked-in models and the Gemini timeout); a repeated question is answered from the cache in about 2 seconds. **Not yet proven**: the online judge against a real model; a full offline evaluation of the golden set (the first attempt on 2026-09-26 stopped when the shared free-tier Gemini quota ran out, so the RAGAS and DeepEval paths have never run for real); and the overlap-merge safety net against a live model. MLflow has never logged a real run. **Known gaps**: some papers have few or no method/dataset/metric links in the graph and no baselines were extracted, so graph-style questions can miss them; tables are retrieved and shown to the model but are not part of citation checking; the citation guardrail can show an "ungrounded_citations" note on a correct answer when the model shortens a paper's file name (seen live); the admin-only "Tool-Calling" pattern in the UI is a local comparison demo whose module is not deployed; the live app and evaluations share one free Gemini quota; and the concurrency limiter is per process (Cloud Run can scale to more than one instance). **Not built**: password-strength rules beyond length, email verification and reset flows, per-IP rate limiting (limits are per user), and answer streaming (deliberately, so every answer passes the output guardrail before the user sees it).
