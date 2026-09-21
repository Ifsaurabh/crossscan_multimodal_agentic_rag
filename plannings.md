# RAG Project Plan

## Where the project stands (paused 2026-09-19 — user took a break and shut the laptop down)

**Working method**: the user asked to work through the "left to build" list **one decision at a time**. Item 1 (multi-LLM fallback) is decided and built (section 12c). **Item 2 (sign-up hardening) was asked and left unanswered** — options: nothing more (recommended: the mandatory global daily cap already bounds abuse cost) / email verification / CAPTCHA / per-IP limits. Next: item 3 shared rate limiting/lockout (Redis), item 4 deployment (TLS, Dockerfile build, Streamlit deployment), item 5 golden set v2, item 6 live guardrail check.

**Decisions taken so far** (all by the user): D1 global daily cap is mandatory (default 50, adjustable); D2 open sign-up with 2 messages/day per regular user, sign-ups throttled 10/hour; D3 install Streamlit (done, 1.64.0); D4 create the app tables (done); D5 create the admin account now with a saved random password in the gitignored `.env` (done; user will change it later); D6 Langfuse prompt sync deferred; D7 tests paused; D8 folder rename to CrossScan is the user's job; LLM fallback order Gemini -> Anthropic -> OpenAI with `claude-haiku-4-5-20251001` and `gpt-4o-mini`; `anthropic` install deferred.

**Verified by running**: ingestion (Stages 1-6b), retrieval (Stages 7-10, 3 live queries), golden set v1, DVC tracking/push, Streamlit install, app-table DDL, admin creation. **Written but never run**: everything in sections 11-12c (chatbot UI, auth, quotas, memory, API, LLM connection, evaluation runner, MLflow, Langfuse client/prompts/tracing, usage/cost tracking) and all their tests. Shared code changed a lot since the last test runs, so the first run will likely surface bugs — run the shared-code tests first (`test_gemini_retry`, `test_agent_transform_route`, `test_agent_quality_generate`, `test_retrieval_graph`, `test_run_query`, `test_llm_connection`), then the rest, then a real-database pass.

**Open items**: install `anthropic`; user adds `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` to `.env`; decide whether `extract_entities.py` and the RAGAS judge should also use `llm_connection`; Langfuse UI reachability + prompt sync (deferred); Neo4j start + first live query + first evaluation run (postponed); user renames the folder; git commit only when asked, after the rename. Full resume checklist: `context_prompt.md`, top section.

## Scope

- **Multimodal**: text + images (figures, charts, CT/satellite imagery) extracted, embedded, and retrievable together.
- **Graph DB**: alongside vector DB, to capture explicit relationships (shared methods, datasets, techniques across papers) that similarity search misses. Hybrid retrieval, not a replacement for vector search.
- **Agentic**: orchestrator/agent decides which retrieval tool to use (vector vs. graph vs. both) and can decompose multi-step questions rather than a single fixed retrieve-then-generate flow.
- **Use case anchor**: a research-assistant tool for literature review — find relevant findings across papers, surface figures/charts in context, and identify shared methodology across domains (e.g. CNN used in both medical and remote-sensing papers).
- **PII scope**: not about the documents (published papers, no patient/personal data) — about user **queries** that may contain PII (e.g. names, emails). Redact at the input boundary before logging/tracing or sending to any LLM API.
- **Testing approach**: two distinct kinds, on different timelines.
  - **Unit tests (pytest)**: deterministic code logic (extraction, guardrails, chunking, etc.) — written **incrementally, right after each component is built**, not deferred to project end. `tests/` at project root, mirrors `src/` modules. Heavy dependencies (Docling conversion, the NSFW vision model) are mocked/faked in tests to keep the suite fast — only I/O-light logic and small real fixtures (tiny in-memory PDFs via pymupdf, tiny PIL images) are exercised directly.
  - **Eval tests (Stage 11)**: qualitative RAG behavior (retrieval relevance, groundedness, hallucination rate) — can only meaningfully exist once Retrieval/Generation are built, so they start once those stages exist, not "at the end."
  - Backfilled for Stages 1-4 in one pass once the gap was noticed (32 unit tests, all passing) — going forward, tests should be written alongside each new stage instead of batched.

## 1. Data Loading

- Done: dataset downloaded via `kagglehub` (`saibhossain/rag-practice`, 12 PDFs) and copied to `data/raw/` via `src/load_data.py`. No Kaggle credentials needed (public dataset).
- Data is fully reproducible from code + Kaggle source — `archive.zip` removed from the project; nothing binary needs to be stored or committed.
- Confirmed: PDFs contain both text and embedded images (not scanned) — no OCR needed for body text.

## 2. Data Cleaning

- Text extraction: **Docling** (layout-aware, handles multi-column academic layout, tables, section structure).
- Image extraction: **PyMuPDF (fitz)** (extracts embedded images with page-level metadata).
- Two separate extraction methods, two separate tools — each used for its strength.

## 3. Document Preparation

## 3b. Ingestion Guardrails

- New stage, separate from stage 7 (Input Guardrails, which checks user queries at retrieval time). This runs on the documents themselves, before chunking.
- Detection method: **heuristic/rule-based** (regex/pattern-based), not LLM-based — fast, free, no external calls.
- **Text** (`src/ingestion_guardrails.py`):
  - **PII**: email redaction only. Phone-number regex was tried and dropped — it false-matched journal citation numbers (e.g. "Procedia Computer Science 105 (2025) 1-8") as phone numbers, corrupting text for no real benefit (research papers essentially never contain phone numbers).
  - **Unsafe content**: keyword-based match → drop the section entirely if matched. 0 hits across the corpus (expected for a research-paper dataset).
- Output: `data/guarded/*.json` (redacted, unsafe sections dropped) + `data/guardrail_report.json`.
- **Images** (`src/image_guardrails.py`): text heuristics can't assess visual content, so this uses a free/local **vision model** instead — Hugging Face `Falconsai/nsfw_image_detection`, run as a batch classifier over all extracted images (no API cost, runs locally). Flagged images (score ≥ 0.5) are dropped from disk and `metadata.json`, same pattern as `quality_check.py`.
  - Result: 0/174 images flagged (expected — corpus is research-paper figures: CT scans, satellite imagery, charts, not photographic content).
  - Output: `data/image_guardrail_report.json`.

## 4. Chunking

- Strategy: **parent-child chunking**, built on top of Stage 3's section structure.
  - **Parent** = section (Methods, Results, etc.). Cap ~1500-2000 tokens; oversized sections split into multiple parent sub-chunks (same section label, sequential parts).
  - **Child** = paragraph within a section, used for embedding/search precision. Cap ~300-500 tokens; oversized paragraphs recursively split (sentences, then fixed windows) until under the cap.
  - **Overlap**: 20% between adjacent chunks (both parent and child level) to avoid losing context at boundaries.
  - Rationale: small chunks embed/match precisely, but lack context alone; parent-child retrieves precisely (child) while returning enough context to answer well (parent). Numbers chosen are within common industry ranges (child 200-500, parent 1000-2000+ tokens, 10-20% overlap).
- **Metadata (core set)** on every chunk: `source_pdf`, `section` heading, `chunk_id`, `parent_id` (children only), `chunk_type` (parent/child), `page_start`/`page_end`. `domain` deliberately **not** included — deferred to the vector-storage/namespace stage rather than decided here.
- **Tokenizer**: `tiktoken` (`cl100k_base` encoding) for measuring chunk sizes against the caps above.
- **Tracking**: no `token_count` stored in chunk metadata (cheap/local to recompute later if ever needed — not worth the bloat). Instead, `chunking_report.json` logs token count distribution (min/max/avg per parent and child) and processing latency per document.
- **Implemented** (`src/chunk_documents.py` + `src/chunking_config.py`): config-versioned as `v1_parent1800_child400_overlap20pct`. Input: `data/guarded/*.json`. Output: `data/chunks/*.json` (nested parent→children) + `data/chunking_report.json`.
  - Parent-child grouping (paragraph packing, overlap carry-forward) is **custom** code. The fallback path (when a single paragraph exceeds the cap on its own) uses **`RecursiveCharacterTextSplitter`** (`langchain-text-splitters` — the lightweight package, not full `langchain`), token-aware via `tiktoken`, instead of hand-rolled sentence/fixed-window splitting.
  - Result: 316 parents, 623 children across 12 documents. All parents ≤1798 tokens (cap 1800). Child chunks mostly ≤400, two documents slightly over (409, 426) — a minor known effect of joining paragraphs/sentences with `\n\n` separators (separator tokens aren't counted during accumulation, only at final encode) — still within the broader intended child range (~300-500 tokens), not a real problem.
  - Runs fast: <0.25s per document (no ML models involved, pure tokenization + grouping logic).
- **Text-only** — chunks do not carry image references. Image-to-text relationships (e.g. "this figure belongs near this paragraph/section") are deferred to Stage 6b (Graph Storage) as an explicit graph relationship, not chunk metadata — keeps chunking simple and puts relationship-modeling where it already belongs.

## 5. Embedding

- **Two separate models, two separate embedding spaces** (not shared) — images are linked to text via metadata/graph DB (Stage 6b), not via direct vector-space proximity. Considered a shared-space single model (SigLIP, for native cross-modal text↔image search) but rejected: SigLIP's text encoder is meaningfully weaker at pure text retrieval than a dedicated text model, and most queries will be text questions — text retrieval quality was prioritized over native cross-modal search.
- **Text**: `bge-base-en-v1.5` (via `sentence-transformers`, free/local). Chosen over `bge-small-en-v1.5` — at this corpus scale (~900 chunks) the larger model's better retrieval quality has no meaningful compute/storage cost. Chosen over OpenAI `text-embedding-3-small` — project has consistently used free/local models (Docling, PyMuPDF, Falconsai NSFW, now embeddings) over paid APIs.
- **Images**: OpenAI CLIP `ViT-B/32` (via `transformers`, free/local, no new dependency needed — already installed). Considered OpenCLIP (LAION-trained, better quality) and SigLIP (best quality-for-size, newer training approach) but user chose the original, most widely-documented CLIP checkpoint.
- System requirements: CPU-only is sufficient for both models at this scale (~2-4GB RAM, no GPU needed) — consistent with the project's existing CPU-only torch setup.
- **Scope**: only **child chunks** are embedded (their stated purpose from Stage 4 — search precision). Parent chunks are stored but not embedded; fetched for context only after a child match at retrieval time.
- **Implemented** (`src/embed_text.py`, `src/embed_images.py`, `src/embedding_config.py`).
  - Text: input `data/chunks/*.json`, output `data/embeddings/text/*.json` (chunk_id, parent_id, source_pdf, section, page range, 768-dim embedding) + `data/text_embedding_report.json`. Result: 625/625 child chunks embedded.
  - Images: input `data/images/metadata.json`, output `data/embeddings/image_embeddings.json` (image_file, source_pdf, page, 512-dim embedding) + `data/image_embedding_report.json`. Result: 174/174 images embedded, 0 failures.
- **Structure-aware embedding** (v2, `v2_bgebase_clipvitb32_structureaware`, supersedes v1 `v1_bgebase_clipvitb32`): child chunk text is prefixed with `"Document: {source_pdf} | Section: {section}\n\n"` before embedding (stored chunk text itself unchanged — only what's fed to the model changes). Rationale: parent-child chunking only improves what's returned *after* a correct match; it does nothing if the wrong chunk is matched in the first place. Short/generic chunks (e.g. "The model achieved 98% accuracy...") can read near-identically across different papers/sections — without document/section context baked into the vector itself, similarity search can't distinguish them. Images not affected (CLIP embeds visual content, not text context).

## 6. Vector Storage

- Store both text and image embeddings, with metadata linking images back to source PDF/page/nearby text.
- **Namespaces**: partitioned by **content type + domain** (e.g. `text-lung-cancer`, `text-land-cover`, `image-lung-cancer`, `image-land-cover`) — separates text/image (different embedding types/dimensions) and lets the agent narrow search scope by topic.
- **Database: pgvector on user's existing Postgres** (not a new service — deployment consideration was decisive: project needs to be deployed for a live demo later, and reusing already-deployed infra beats introducing a new one). Considered Qdrant (purpose-built, easy local→cloud migration path) and FAISS (ruled out — no deployment/serving story, just a library) — at this project's scale (625 text + 174 image vectors) none of the vector-DB-specific performance differences matter, so deployment simplicity won.
- **Hybrid search** (vector + keyword, for later): native Postgres full-text search (`tsvector`/`tsquery`) combined with pgvector similarity in the same SQL query (e.g. reciprocal rank fusion) — no extra service needed, same database.
- **Domain tagging** (deferred from Stage 4, now resolved): **embedding-similarity classification** (`src/classify_domain.py`), reusing the `bge-base-en-v1.5` embeddings already computed — no new model.
  - Seed domains (`lung-cancer`, `land-cover`) defined via short label descriptions, embedded once. Each document's chunk embeddings are averaged into a document vector and compared via cosine similarity.
  - **Dynamic domain discovery**: if no seed domain scores above threshold (0.60, chosen from observed data — correct matches scored 0.66-0.76, the one true mismatch scored 0.53), a **new domain is created** instead of forcing a wrong classification, named via a slug of the document's title.
  - **A local LLM (Qwen2.5 3B via Ollama) was tried and rejected**: asked to fully replace the embedding classifier (sequential prompting, growing domain list per document). Produced worse results — fragmented the same domain into multiple names (`satellite-image-analysis` vs `remote-sensing`; `ai-powered-lung-cancer-detection` vs `intelligence-based-medicine`), returned "none" as a non-answer for 2 documents, and **misclassified a lung-cancer paper as `ai-security`**. The embedding approach classified all 12 documents correctly; Qwen did not. Root cause: a 3B quantized model isn't reliable enough for consistency-dependent sequential classification.
  - **Bonus bug fix found in the process**: title extraction (`get_document_title`) could pick up journal front-matter labels ("OPEN ACCESS", "REVIEWED BY", etc.) as the title instead of the real one, since Docling tags them as `section_header` blocks appearing before the actual title. Fixed by skipping a known front-matter label set. This only affects the new-domain-naming path (embedding classification itself is robust to it, since it uses whole-document averaged embeddings, not the title) — but could matter later for citations, so worth remembering.
  - Result (12/12 correct): 10 documents split cleanly into `lung-cancer` / `land-cover`; the 1 true outlier (an AI/LLM security paper, unrelated to either domain) correctly got its own new domain rather than being force-fit.
  - Output: `data/domain_classification.json` (source_pdf → domain) + `data/domain_classification_report.json` (scores, which documents triggered new-domain creation).
- **Implemented** (`src/db.py`, `src/setup_vector_db.py`, `src/load_vector_db.py`):
  - Driver: **`psycopg` v3** (not `psycopg2`, which the user's other project uses) — chosen for future async support if a high-concurrency API server is built later (Stage 12), even though sync vs async makes no difference at the current batch-script stage. Plus the `pgvector` Python package for vector type adapters.
  - Database: a **brand-new dedicated database** (`new_rag_0926`) on the user's existing local Postgres server, fully separated from other projects (techchefz, My CEO) — not a shared database, own dedicated **schema** (`rag_new`) too.
  - Tables: `text_parents` (full parent text, not embedded), `text_chunks` (child text + `vector(768)` embedding + domain), `images` (`vector(512)` embedding + domain). HNSW indexes (cosine distance) on both embedding columns; btree index on `domain` for namespace-style filtering.
  - Namespace design realized as **domain column + separate tables per content-type** (not physical Postgres namespaces) — the relational equivalent of the original text/image + domain partitioning plan.
  - **Real bug found and fixed**: `CLIPModel.get_image_features()` in the installed `transformers` version (5.17.0) returns the wrong object — a raw `BaseModelOutputWithPooling` (vision encoder output) instead of the documented projected 512-dim embedding, with no `image_embeds` attribute to recover it. This silently produced corrupted 50-dimensional embeddings (undetected until the pgvector insert failed with "array must be 1-D", since the corrupted shape was nested and never dimension-checked before). Fixed by manually calling `model.vision_model(...)` + `model.visual_projection(...)` — the two steps `get_image_features()` is supposed to perform internally. All 174 images re-embedded correctly (verified 512-dim) before reloading into the DB.
  - Result: 316 parents, 625 text chunks, 174 images loaded. Verified end-to-end with a real cosine-similarity query (top match was the query chunk itself at 1.0 similarity, next closest a related same-domain paper).

## 6b. Graph Storage

- Entity/relationship extraction (methods, datasets, techniques, domains) for graph queries alongside vector retrieval.
- **Image-text relationships**: captures which images relate to which paragraph/section (e.g. caption/figure mentions) as an explicit graph edge, since chunking (Stage 4) deliberately stays text-only and doesn't embed image references into chunks.
- **Entity extraction — tried Qwen2.5 3B (Ollama) first, then switched to Gemini after confirmed hallucination.**
  - Scope: per-document (not per-section — would take 16+ hours), using Abstract + Conclusion text, extracting `methods`/`datasets`/`metrics` as three categories.
  - **Qwen2.5 3B**: single-document test looked good, but the full 12-document run showed **confirmed hallucination on 2/11 non-skipped documents** — one invented specific unstated numbers (fabricated YOLOv12/v13/v14 when the source only said "the other four YOLO versions" without naming them), the other fabricated an entirely unrelated methods list (CNN/YOLOv11/DWT) for a paper whose text never mentions any of them. Also very slow (204s cold-start, 20-80s per call after).
  - **Switched to Google Gemini** (`gemini-3.6-flash`, official `google-genai` SDK) — a free *hosted* API rather than free *local*, a deliberate category shift (data leaves the machine, but these are just published-paper abstracts, no privacy concern). Checked all of the user's other local projects for an existing API key first — none found, user obtained a fresh one from Google AI Studio.
  - **Prompt hardened** with explicit anti-hallucination rules ("only extract what's explicitly named," "do NOT infer/guess unstated specifics") directly targeting the failure mode found with Qwen.
  - **Added a verification step**: after extraction, checks whether each entity's exact text appears in the source (case-insensitive substring match), splitting results into `verified`/`unverified` rather than trusting everything blindly. Catches likely fabrication automatically, though it also flags legitimate paraphrases (e.g. "CNN" in source vs "Convolutional Neural Network" in output) as false positives — a strict but honest signal, not perfect precision.
  - **Real bug found and fixed in the process**: an early "resume" feature (added to survive the quota/rate-limit crashes) initially loaded stale output files from the *previous* Qwen-based run and incorrectly treated them as already-completed Gemini work, silently skipping all 12 documents on a retry. Fixed by checking the saved report's `model` field matches the current model before trusting it as resumable state.
  - **Final result (12/12 documents processed, complete)**: noticeably better quality than Qwen throughout — richer extraction (more methods/metrics per paper). Hit the free-tier daily quota (20 requests/day) after 10/12; the remaining 2 finished the next day after the quota reset (session resumed after a machine shutdown in between — see the "session resume" pattern this project now has via `context_prompt.md`).
    - `fmed-12-1567119.pdf` (previously fabricated YOLOv12/v13/v14 with Qwen) now correctly extracts only `['YOLO', 'YOLOv8']` — no invented version numbers.
    - `Utilizing Sentinel-2...pdf` (previously got an entirely fabricated methods list with Qwen) now correctly returns **0 methods, 0 datasets, 0 metrics** — Gemini recognized the abstract/conclusion genuinely names no specific ML methods (the paper is about LULC/NDVI percentages) and returned an honest empty result instead of inventing content. This is the clearest possible confirmation the hallucination problem is fixed: faced with the exact document that broke Qwen, Gemini said "nothing here" rather than fabricating.
  - Output: `data/entities.json` (`{source_pdf: {verified: {...}, unverified: {...}}}`), `data/entity_extraction_report.json`. `src/extract_entities.py`.
- **Database: Neo4j** (separate service from Postgres/pgvector). Considered consolidating onto Postgres via the Apache AGE extension (one database for vectors + hybrid search + graph — simpler deployment) but chose a dedicated graph DB instead — Neo4j has far more mature tooling/documentation/community than Apache AGE, worth the extra service for a better-supported learning experience. Re-confirmed over Memgraph/ArangoDB/NetworkX when asked for a broader comparison.
- **Setup**: no existing Neo4j installation found (checked all other local projects too — none use it). Ran via Docker (`docker run -d --name neo4j-rag-new -p 7474:7474 -p 7687:7687 -v neo4j_rag_new_data:/data neo4j:latest`) — Docker Desktop wasn't running initially and had to be started manually; no Java installed either, ruling out a standalone install.
- **Password set programmatically, not via browser**: user was working remotely from mobile and couldn't reach `localhost:7474`. `neo4j-admin dbms set-initial-password` was tried first but doesn't work post-startup (only takes effect before first launch — confirmed by its own warning message). Fixed by connecting via the `neo4j` Python driver with default credentials and running `ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO '<new password>'` in the `system` database — the correct way to change a password on an already-running instance. Verified with a test query.
- Connection details (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`) added to `.env`/`.env.example`. `neo4j` Python driver added to `requirements.txt`.
- **Schema designed and implemented.**
  - Nodes: `Paper` (source_pdf, domain), `Method`, `Dataset`, `Metric` (all deduplicated/shared across papers — a "CNN" mentioned in two papers is one node, not two), `Section` (chunk_id, heading, page_start, page_end), `Image` (image_file, page).
  - Relationships: `(Paper)-[:USES_METHOD]->(Method)`, `(Paper)-[:USES_DATASET]->(Dataset)`, `(Paper)-[:EVALUATED_WITH]->(Metric)`, `(Paper)-[:HAS_SECTION]->(Section)`, `(Image)-[:APPEARS_IN]->(Paper)`, `(Image)-[:NEAR_SECTION]->(Section)`.
  - **Image-text linking**: no new extraction needed — uses existing metadata (image's `page` vs. each section's `page_start`/`page_end`), matched by simple overlap. 174/174 images (100%) successfully linked to a section this way.
  - **Entity source**: only `verified` entities from Stage 6b's Gemini extraction are loaded (not `unverified`) — the unreliable/unconfirmed ones are deliberately excluded from the graph.
  - `src/graph_db.py` (driver helper), `src/setup_graph_db.py` (uniqueness constraints on all 6 node types), `src/load_graph_db.py` (loads papers, entities, sections, images).
  - **Result**: 12 Paper nodes, 70 USES_METHOD / 7 USES_DATASET / 22 EVALUATED_WITH relationships, 316 Section nodes, 174 Image nodes (174/174 linked to a section).
  - **Validated the actual use case**: a cross-paper shared-method query found `"CNN"` used by both a lung-cancer paper and a land-cover paper — flagged as cross-domain — directly matching the project's original use-case anchor ("identify shared methodology across domains"). This is the kind of query vector similarity search can't do cleanly; the graph traversal finds it directly.

## 7-10. Retrieval Pipeline — Full Agentic Design (finalized, NOT YET BUILT)

Extensive real-time design discussion (deliberately "discuss only, don't build" per user instruction throughout). Everything below is agreed conceptually; **zero code exists for any of this yet**. Stage 8 (Retrieval) was designed before Stage 7 (Input Guardrails) — user's choice, on the reasoning that guardrails are easier to design once it's clear what they protect.

### Architecture: 2 agents + deterministic tools, not a single agent or 6 agents

Considered two shapes: (a) one ReAct-style agent freely choosing tools, vs (b) an explicit multi-node graph (LangGraph-style) with fixed structure. **Chose (b)** — the flow has specific ordering requirements (e.g. cache check must always run first, deterministically) that a freely-reasoning single agent isn't guaranteed to respect. Started at 4-5 agents (one per job: transformation, routing, quality-judge, generation, +guardrails), **reduced to 2 agents** after review — not every step needs LLM judgment, and guardrails specifically should NOT be agents at all (see below).

**Why guardrails are deterministic Tools, not Agents (key insight, drove this decision)**: an LLM-based guardrail would require sending the (potentially sensitive) query to an LLM just to check whether it's safe to send to an LLM — self-defeating. This matches how Stage 3b (Ingestion Guardrails) was already built — plain regex, not an LLM call. Same principle: **Input Guardrails (Stage 7) = deterministic Tool** (reuses the email-regex pattern from `ingestion_guardrails.py`), and it must run as the **very first step in the whole flow**, before even the cache check — redact PII before it's used as a cache key or sent anywhere. **Output Guardrails (Stage 10) grounding check** should also be codeable, not an agent — reuse the same "verify entity text appears in source" pattern already built and proven in Stage 6b's entity extraction (check whether each generated claim's citation corresponds to actually-retrieved content).

### The 2 Agents

1. **Transform + Route Agent** — combines query transformation and routing in one agent:
   - **Decomposition**: splits compound questions into sub-queries. Count is dynamic, decided by the LLM per query (not fixed).
   - **Expansion**: generates up to **3** phrasing variants per sub-query (capped, to bound the call/retrieval multiplier — uncapped expansion was rejected as too expensive).
   - **Routing**: runs *per sub-query* (not once for the whole compound question — a shared decision would misroute mixed cases like "which papers use YOLOv11 and what's its performance," where one sub-query needs graph and the other needs vector). Outputs a structured decision per sub-query:
     - `needs_retrieval` (bool) — if false, answer from general knowledge, response **must be labeled "not from the knowledge base, AI-generated"** (a transparency/trust feature — a research tool's users need to know when an answer isn't grounded in the actual corpus).
     - `data_source`: `vector` / `graph` / `both`
     - `search_mode`: `simple` / `hybrid` (semantic + keyword) — **vector-only**. Considered "hybrid" for graph search too but rejected: the graph has no semantic/embedding signal to combine with anything (Method/Dataset/Metric nodes are matched by exact name, not embedding similarity) — "hybrid" only means something where two independent signals (semantic + keyword) exist for the same content, which is true for vector search, not graph traversal as currently built.
     - `images_required` (bool) — independent of `data_source`. Judged on query intent ("show me...", "what does X look like...") vs. pure fact/relationship questions that don't need visuals. When true, images are **not** an independent search — they're resolved via the graph's existing `Image-[:NEAR_SECTION]->Section` / `Image-[:APPEARS_IN]->Paper` relationships, scoped to whichever sections/papers the main retrieval already touched (reuses Stage 6b's schema entirely, no new infrastructure).
   - **On retry** (see below), receives structured feedback from Agent 2 (what was missing, what to look for) and re-routes *informed* by that diagnosis, not a blind retry.

2. **Quality Check + Generate Agent** — judges retrieved results, then either generates or triggers a retry:
   - If retrieval is sufficient → generates the final answer directly (inline per-claim citations, e.g. "[LungPaper.pdf, p.4]" — decided earlier in Stage 9's original scoping).
   - If insufficient → triggers the retry sequence (below) instead of generating immediately.
   - Vision-capable generation needed when images are part of the context (Stage 9's original requirement, still holds).

### Retry policy (bounded, max 3 attempts total)

```
Attempt 1: Agent 1 routes → Retrieval → Agent 2 judges
    │ insufficient
    ▼
Attempt 2: Reranker (local, free cross-encoder model) applied → Agent 2 re-judges
    │ still insufficient
    ▼
Attempt 3: Agent 2 sends structured feedback (what's missing + what to look for)
           → Agent 1 re-routes, informed by that feedback → Retrieval → Agent 2 judges
    │ still insufficient after 3 attempts
    ▼
Generate best-effort answer anyway, with a transparency caveat
("retrieval confidence was low, answer may be incomplete")
```

No infinite loop risk — hard cap at 3 attempts, then honest fallback rather than looping forever. Reranker is invoked conditionally (Agent 2's call), not on every query — cheaper attempt (no new retrieval) tried before the more expensive re-routing attempt.

### Combining `both` results (vector + graph)

- **Text**: for **complex/decomposed queries**, narrow the vector search using the graph-filtered paper list (e.g. graph finds "papers using CNN," then vector search is scoped to only those papers' chunks) — more precise than searching everything.
- **Images**: the same paper-level narrowing does **NOT** apply cleanly to images (a paper matching on entity X doesn't mean all its images are about X — too coarse a filter). Images stay a separate, unnarrowed lookup via the graph relationships described above, not filtered by the same paper list used for text narrowing.

### Caching (Postgres, not Neo4j)

Considered storing the cache *in* Neo4j (queries/chunks/answers as graph relationships) — rejected as the primary cache mechanism: Neo4j is optimized for relationship traversal, not fast point lookups, so it would make the cache check itself slower than the thing it's meant to speed up. **Decided: conventional Postgres table**, storing every query + its retrieved chunks + the generated answer (doubles as both cache and query log). Two cache checks in the flow:

```
Query → [Input Guardrail: tool, redact PII FIRST] → [Cache Check #1: raw query, Postgres lookup]
                                                    │
                                                    ├── HIT → return immediately
                                                    │
                                                    └── MISS
                                                          │
                                                          ▼
                                              [Transform + Route Agent]
                                                          │
                                                          ▼
                                              [Cache Check #2: transformed query, Postgres lookup]
                                                          │
                                                          ├── HIT → return immediately
                                                          │
                                                          └── MISS
                                                                │
                                                                ▼
                                              [Retrieval Executor: tool] → [Quality Check + Generate Agent]
                                                                │
                                                                ▼
                                              [Cache/Log Writer: tool] → store under BOTH raw + transformed query keys
```

Two checks (raw query, then transformed query) give both benefits without extra cost: the raw check stays free/deterministic and catches exact repeats with zero LLM cost; the second check (after transformation) catches semantically-equivalent-but-differently-phrased repeats, but only pays the transformation cost on an actual miss.

**Prompt caching: attempt opportunistically, fall back gracefully — resolved.** Checked Gemini's actual pricing — context/prompt caching costs $0.15/1M cached tokens read **plus** a $1.00/1M tokens/hour storage fee, accruing continuously even when idle; no confirmed free-tier path. Initially decided to skip entirely, but reconsidered: Gemini's caching isn't a simple flag, it's an explicit two-step API (`cachedContents.create` first, then reference the cache ID in generation calls) — so the implementation can **try to create the cache, and on any failure (no billing enabled, quota, etc.) just fall back to a normal uncached generate call**. Worst case if unavailable: one extra failed API round-trip (negligible latency, no cost) — nothing else breaks. Best case if it ever becomes available (billing enabled later, terms change): automatic savings with no code change needed. **Decision: implement with this try/fallback pattern** for the Transform+Route and Generate agents' fixed instructional prefixes, rather than hard-skipping the capability.

### Deterministic Tools (not agents)

- **Input Guardrail** — PII regex/redaction on the query, runs first, before anything else touches the query.
- **Cache Check** (×2) — Postgres lookup by query text/hash.
- **Retrieval Executor** — runs the actual DB calls (pgvector similarity, Postgres hybrid full-text, Neo4j Cypher traversal, image lookup via graph) per whatever Agent 1 decided.
- **Reranker** — **`BAAI/bge-reranker-base`** (local, free — same model family as the `bge-base-en-v1.5` embedding model already in use, consistent tooling).
- **Cache/Log Writer** — persists query + chunks + answer after a fresh run.
- **Output Guardrail** (grounding check) — verify-in-source pattern reused from Stage 6b, not yet built.

### Framework: LangGraph (finalized)

Chosen for the explicit, ordered multi-node structure this design needs (agent nodes + tool nodes with fixed sequencing, e.g. cache check must always run first) — a fit confirmed after the "single free-form agent vs. explicit graph" comparison above.

### IMPLEMENTED AND TESTED

Every piece of the design above is now built:
- `src/retrieval_config.py` — model names, retry/expansion caps.
- `src/query_guardrail.py` (Stage 7, Input Guardrail — deterministic, PII regex + prompt-injection detection + medical-advice framing detection, reuses Stage 3b's pattern). User explicitly asked for prompt-injection coverage mid-build, added to both Stage 7 (input) and Stage 10 (output, defense-in-depth).
- `src/query_cache.py` (Postgres `query_cache` table — exact-match cache by query hash, doubles as query log).
- `src/retrieval_executor.py` — `vector_search`, `hybrid_vector_search` (added a `tsvector` generated column + GIN index to `text_chunks` for this — the "for later" hybrid-search idea from Stage 6 is now actually built), `graph_search` (substring entity-name matching against query text, traverses to Papers), `get_images_for_sections` (reuses the `NEAR_SECTION`/`APPEARS_IN` graph relationships from Stage 6b).
- `src/reranker.py` — `BAAI/bge-reranker-base` via `sentence_transformers.CrossEncoder`.
- `src/agent_transform_route.py` (Agent 1) and `src/agent_quality_generate.py` (Agent 2) — both call Gemini via `src/gemini_retry.py`, a shared retry-with-backoff helper (same pattern proven in `extract_entities.py` — Gemini's occasional 503s under load needed this in practice, confirmed during the real end-to-end test below).
- `src/output_guardrail.py` (Stage 10) — citation verification (reused the Stage 6b "verify entity text in source" pattern), prompt-injection-leak detection, clinical-overstatement detection.
- `src/retrieval_graph.py` — the LangGraph `StateGraph` wiring all of the above into the full designed flow (cache ×2, retry loop with reranker/re-route branches, all as designed).
- `src/run_query.py` — a runnable CLI entry point (`python run_query.py "<question>"`).

**Real bug found and fixed during testing**: the citation regex only matched single-page citations (`[paper.pdf, p.4]`), but Gemini's actual generation output sometimes produces multi-page citations (`[paper.pdf, p.1, p.14]`) — silently failing to verify these instead of flagging them. Found via a live end-to-end run, not a unit test (the mocked tests all used single-page citations). Fixed by changing the citation regex to capture a page-number *group* and extracting all page numbers within it.

**Testing**: 62 new pytest tests across all the files above (query_guardrail, query_cache, retrieval_executor implicitly via integration, reranker, agent_transform_route, agent_quality_generate, output_guardrail, gemini_retry, retrieval_graph), all passing.

**Live end-to-end verification** (not just mocked unit tests): ran a real query ("What accuracy did the lung cancer CNN model achieve?") through the actual compiled LangGraph, hitting real Postgres, real Neo4j, and real Gemini. Result: a correctly grounded, multi-source answer with accurate inline citations pulling real numbers from multiple papers (e.g. "Enhanced CNN: 100% testing accuracy [Deep learning-based approach...pdf, p.1, p.14]"), and the retry-with-backoff logic visibly recovered from several transient Gemini 503s during the run without failing.

**Prompt caching: implemented, tested, and live-verified.** `gemini_retry.py` now has `call_with_cache()` — attempts `client.caches.create()` once per distinct (model, system_instruction), remembers success/failure in a module-level registry (never retries a doomed cache-create call on every request), and falls back to inlining the instructions if caching isn't available. Both agents (`agent_transform_route.py`, `agent_quality_generate.py`) were restructured to split their prompts into a **fixed instructional block** (the cacheable `system_instruction`) and **variable content** (query/context/feedback — never cached, since it's different every call). 8 new tests (6 for the caching mechanism itself, plus fixture updates in both agent test files) — 44 total tests across the Stage 7-10 files, all passing.

**Real finding from live testing — caching is rejected for a concrete, specific reason, not billing**: `400 INVALID_ARGUMENT: Cached content is too small. total_token_count=477, min_total_token_count=1024`. Gemini requires a **minimum 1024 tokens** in cached content; our instruction blocks (108-477 tokens) don't meet it. This is a genuinely different (and more specific) constraint than the earlier billing-uncertainty assumption. The fallback handled it perfectly — no crash, clean message, query completed successfully with a correctly-grounded cross-domain answer (found 4 papers using CNN across both medical and land-cover domains, exactly the project's anchor use case). If the instruction blocks ever grow past 1024 tokens (e.g. adding few-shot examples), caching would activate automatically with zero code changes needed — the mechanism is already correct, just currently below Gemini's size floor.

### Transparency-label bug: found and fixed
While auditing remaining work ("what next, show me list"), found a compounding bug in the `needs_retrieval: false` path — the one place general-knowledge sub-queries (no retrieval needed) are supposed to be answered and labeled "not from the knowledge base, AI-generated" (designed early in the Stage 7-10 discussion). Two problems:
1. The label was never actually implemented — `needs_retrieval: false` sub-queries fell through to the normal `generate_answer()` with empty chunks, with no disclosure of any kind.
2. They were also wastefully going through the full quality-check/retry loop — `check_quality()` on an empty chunk list always returns `sufficient: False`, so these sub-queries silently burned 2-3 pointless LLM calls (judgment, reranker no-op, re-route) before eventually falling back.

Fixed in `agent_quality_generate.py` (added `GENERAL_KNOWLEDGE_SYSTEM_INSTRUCTION`, `NOT_FROM_KNOWLEDGE_BASE_LABEL`, `generate_general_knowledge_answer()`) and `retrieval_graph.py` (`node_quality_check()` now short-circuits `needs_retrieval=False` sub-queries to `sufficient=True` immediately; `node_generate()` routes them to the new general-knowledge function instead of `generate_answer()`). 3 new tests, all passing (27 total across the two affected files). **Live-verified**: query "What is CNN?" correctly skipped retrieval entirely and returned an answer prefixed with the transparency label. Per user's scope decision, this closes the label gap; broader Output Guardrails re-verification (injection-leak, clinical-overstatement against live output) is deliberately deferred ("verify later").

### Still open / not yet decided
- Output Guardrails: injection-leak and clinical-overstatement checks implemented (deterministic, not an agent) but not individually re-verified against real live output — deliberately deferred by user ("just fix the label gap for now, verify later").
- No evaluation framework exists yet to systematically test retrieval/generation quality across many queries (Stage 11) — three live queries have now been manually verified (vector-only, graph+vector, and general-knowledge), not a systematic suite. See Section 11 below — now in progress.

### Original Stage 7/9/10 scoping (superseded in shape by the above, content still relevant)
- Medical-advice framing checks (corpus includes lung-cancer research; guard against the system being used as a diagnostic/treatment advisor) — still needed, folded into Input Guardrail tool.
- Off-topic / prompt-injection filtering before queries reach the agent — still needed, folded into Input Guardrail tool.
- Avoid overstating research findings as clinical fact — still needed, folded into Output Guardrail tool.

## 11. Evaluation

**Status: golden-set v1 done (12 examples, one per paper); Synthesizer code built for future automated runs; RAGAS metrics integration not yet started.**

### Golden-set generation
- **Golden-set generation: DeepEval's `Synthesizer`** — user's decision ("Use deepeval for golden set"), made directly.
- **Evaluation metrics (separate from golden-set generation): both DeepEval AND RAGAS** to be used — user: "for evaluation we have to use both, deepeval and ragas, but only for golden set we can choose one."
- Scoping decisions (via AskUserQuestion): input granularity = **parent chunks** (one per paper, the longest/most content-rich, ensuring every paper contributes); size = **~20-30 examples** initially (later reduced to 1/paper = 12, see below); LLM = **Gemini**, wrapped as a custom `DeepEvalBaseLLM` (`src/deepeval_gemini_model.py`) since the project has no OpenAI key — DeepEval's custom-model contract is simpler than expected: `generate(prompt)` with no `schema` param, so DeepEval's own TypeError-triggered fallback path handles JSON parsing itself.
- Storage: **JSONL, git-tracked** (user's explicit request) — `data/golden_set.jsonl`, one JSON object per line: `{input, expected_output, context, source_pdf, domain}`. Not covered by any `.gitignore` pattern (verified).
- `deepeval`, `ragas`, and `langgraph` (previously missing) added to `requirements.txt`.

### Real quota constraint hit (same class of issue as entity extraction earlier)
Live run of `src/generate_golden_set.py` hit Gemini's free-tier daily quota (`RESOURCE_EXHAUSTED`, 20 requests/day for `gemini-3.6-flash`) after the very first context — DeepEval's `Synthesizer` is request-hungry: input-generation, a quality-filter retry loop (up to `max_quality_retries=3`, 1-2 calls each), an evolution pass, and expected-output generation are all separate LLM calls, and by default the filtration "critic model" call also reuses our Gemini model. Worked out the actual math (read `Synthesizer`'s source, not guessed): ~7 calls typical / ~17 worst-case per paper at 2 goldens/paper, ~4 typical / ~6 worst-case at 1 golden/paper.

**Fixes made to `generate_golden_set.py`** (script rewritten, not just config-tweaked):
- `MAX_GOLDENS_PER_CONTEXT` reduced to **1** (fits ~3-4 papers/day instead of ~1-2).
- **Incremental, resumable processing**: one paper at a time, appends to `golden_set.jsonl` immediately after each paper succeeds (previously collected everything and wrote once at the end — today's quota exhaustion would otherwise have lost all progress).
- **Resume-on-rerun**: `load_completed_source_pdfs()` reads the existing JSONL and skips papers already done, so a rerun tomorrow (after the daily quota resets) continues instead of redoing work — same pattern that got entity extraction through its own quota wall earlier in the project.
- Catches `google.genai.errors.ClientError` and checks for `RESOURCE_EXHAUSTED` specifically — stops cleanly with a clear message rather than crashing; any other `ClientError` still propagates.
- 8 new/rewritten tests, all passing (load/skip/incremental-write/quota-stop/non-quota-reraise/noop-when-done).

### Golden-set v1: hand-authored bootstrap (this session)
Rather than wait out the multi-day quota-limited automated run, generated the first golden set manually: pulled the actual representative parent-chunk text per paper straight from Postgres, read all 12 (one per paper — same selection query the automated script uses: `DISTINCT ON (source_pdf) ... ORDER BY length(text) DESC`), and hand-wrote one grounded question + answer pair per paper, in the exact same JSONL schema the automated script produces (`input`, `expected_output`, `context`, `source_pdf`, `domain`). All 12 papers represented, spanning all three domains (lung-cancer, land-cover, the outlier AI-security paper). Written to `data/golden_set.jsonl`, confirmed untracked-but-not-ignored (ready to commit whenever the user asks).

The `Synthesizer`/`generate_golden_set.py` code stays in the project for future fully-automated runs (e.g. once quota allows, or to regenerate/expand the set with more examples per paper) — this was an explicit user instruction: keep the Synthesizer code, but bootstrap v1 by hand for now.

### Golden set v2 (done): 36 hand-written examples
In the "open points" walkthrough the user chose to **hand-write more golden examples instead of using the Synthesizer** (which spends 4-7 model calls per example), **2 more per paper (36 total)**. Method: queried Postgres for two additional parent chunks per paper (preferring Results/Methods/Discussion/Conclusion-type sections, 1.5-6.5k characters, excluding the chunk v1 used), read them, and wrote one question + answer per chunk, each answer grounded only in that chunk's text. Result: `data/golden_set.jsonl` = 36 records, exactly 3 per paper, all 12 papers and all 3 domains. Data-quality notes: one Bhutan chunk labelled "4. Conclusions" was actually a reference list and was replaced by that paper's LULC-statistics and accuracy-assessment chunks; two section labels contained non-breaking spaces (normalized in the build script); one source paper contradicts itself on EfficientNetB0's accuracy (97.9% vs 76.9%), so no question relies on that number. The build script asserted it found exactly the 12 v1 records and all 24 sources before writing (it did catch the non-breaking-space mismatch on the first try and wrote nothing). **Versioning**: `dvc add` + `dvc push` done — DVC md5 changed from v1 `323fe8a7…` to v2 `970985a8a30301845e643d1c1c434a15` (185,768 bytes); MLflow runs record this hash as `golden_set_md5`, so any earlier eval run is distinguishable. The `.dvc` pointer change is not git-committed. **Caveat**: the answers were written by me (an LLM) from the chunks, not by a domain expert, and are single-chunk questions — they test retrieval + grounded answering, not multi-paper synthesis.

### Versioning and MLOps tooling (decided this session; built, not yet run)
User first asked for "versioning", then rejected a hand-rolled `versions.json` registry: **for a demonstration project the tools should be the ones used in industry**. Scrapped it (files deleted, `generate_golden_set.py` reverted, re-tested) and chose:
- **Golden set → DVC.** `dvc init`, local folder remote (`../dvc-storage/crossscan`, outside the repo), `dvc add data/golden_set.jsonl` (md5 `323fe8a767aeb7bc6138c0b5a6d33cc3`), pushed. Git tracks only `data/golden_set.jsonl.dvc`. **Done and run.** The DVC md5 is the golden-set version used in eval runs. Nothing is committed to git yet (files are staged/untracked).
- **Prompts → Langfuse Prompt Management.** `src/langfuse_client.py` (shared client, no-op when disabled or keys missing, never raises), `src/prompt_registry.py` (`get_prompt(name, fallback)` fetches the `production`-labelled version, falls back to the local constant on any failure; `sync_prompts()` pushes local prompts as new versions only when content changed). Prompts managed: `transform-route`, `quality-check`, `generate-answer`, `general-knowledge`. The regex guardrails are config, not prompts, so they are not managed. `agent_transform_route.py` and `agent_quality_generate.py` now call `get_prompt`. **Sync against the live instance not run yet** (web UI was still starting; Docker was slowing the machine).
- **Tracing → Langfuse.** `run_query.invoke_graph()` passes a Langfuse LangGraph callback handler (empty list when disabled). API keys copied from `langfuse/.env` into root `.env` (`LANGFUSE_PUBLIC_KEY`/`SECRET_KEY`/`HOST`).
- **Eval-run tracking → MLflow**, local file store `mlruns/` (gitignored). `src/experiment_tracking.py`: `collect_versions()` (golden md5, per-prompt content hashes, all `retrieval_config` constants) and `log_run()`.
- `tests/conftest.py` now sets `LANGFUSE_DISABLED=1` so unit tests never touch a Langfuse server.
- `langfuse` SDK 4.15.4 installed (small install; also added `wrapt` and three OpenTelemetry/Google common packages). `dvc`, `mlflow`, `langfuse`, `fastapi`, `uvicorn`, `httpx` added to `requirements.txt`.

### Evaluation runner (`src/run_evaluation.py`)
Runs each golden question through the real graph and scores it. **Always-on deterministic metrics (no LLM, free, scale to any size):** `source_hit` (did retrieval touch the expected paper), `citation_verified_rate`, `is_grounded`, `guardrail_flag_count`, `answer_chars`, `latency_s`, plus per-run token usage. **Opt-in LLM-judge metrics (`--llm-metrics`, spend Gemini quota):** DeepEval faithfulness/answer relevancy/contextual precision/contextual recall via `GeminiDeepEvalModel`; RAGAS faithfulness/context precision/context recall via `llm_factory(..., provider="google")`. **RAGAS + Gemini is the least certain piece**: RAGAS routes google-genai clients through `instructor`, and the installed `instructor` is 1.3.2 (pulled in by deepeval), which may be too old — verify on first real use. Each run logs one MLflow run. Metric choices were made autonomously ("no check from now onwards").

### Still open
- Run everything: all new tests, first live evaluation (needs Neo4j, Postgres, Gemini), Langfuse prompt sync, first MLflow run. **User asked to skip running tests for now** (machine slow from Docker) — tests are written, not run.
- Automated Synthesizer run for a larger/v2 golden set: blocked on Gemini quota, resumable via the incremental script.
- Langfuse web/worker had not been verified reachable at localhost:3000 by the end of this session.

## 12. API Integration

**Superseded by the "12b. Chatbot, memory and access control" section below: the stateless, unauthenticated `POST /query` described in this first cut was replaced by authenticated `POST /chat`.** First-cut notes follow. **Built, not run.** FastAPI app in `src/api.py`: `POST /query` (validated 1–2000 chars; returns answer, blocked/block_reason, cache_hit, guardrail flags, latency), `GET /health`, `GET /stats`. Graph is built once, lazily. Blocked queries return 200 with `blocked: true` rather than an HTTP error. `tests/test_api.py` written (FastAPI `TestClient`, graph and invoke monkeypatched).

## 12b. Chatbot, memory and access control

**Why this exists**: after the retrieval/eval/serving work the user pointed out two gaps: (1) **no chatbot** was ever built (only a stateless single-query pipeline/API) and (2) **no memory system at any stage**. Asked to build: Streamlit chat UI (chosen from Streamlit / Chainlit / Gradio / Open WebUI), conversation memory + rolling summary + long-term user memory (all three selected), and — raised by the user as a cross-cutting need "not for memory but access for other things as well" — **authentication, authorization, quotas and limits** so conversations never get mixed up between users.

**Built (all written; NOTHING run — user said write tests but don't run anything; app tables not yet created; Streamlit not installed):**
- `src/setup_app_db.py`: tables `users`, `auth_sessions`, `chat_sessions`, `chat_messages`, `usage_daily`, `user_memories` (pgvector 768-d). Everything user-owned cascades from `users`, so deleting a user erases their data.
- `src/auth.py`: scrypt password hashing (stdlib, no new dependency, params stored in the hash), random login tokens stored only as SHA-256 with expiry (`AUTH_TOKEN_TTL_HOURS`, default 24), generic login errors + dummy hash for unknown users (no timing/enumeration difference), in-memory per-username lockout (5 failures / 15 min), roles `user`/`admin`, admin `update_user` with a column whitelist, deactivation and password reset revoke tokens.
- `src/quotas.py`: per-user daily requests + tokens (role defaults `USER_*`/`ADMIN_*` env vars, per-user DB overrides, NULL = default), sliding-window per-minute `RateLimiter`, `GLOBAL_DAILY_REQUEST_CAP` across all users (protects the upstream quota). **Decision 1 (user): a limit is mandatory but "could be higher" since they can add credits or another LLM -> ON by default at 50/day (0 disables it explicitly); on Gemini's free tier ~5 is what actually fits.** Multi-provider LLM support (a fallback/second LLM) was raised in passing and is not built: all model calls go through `gemini_retry.py` with `GEMINI_MODEL` hardcoded in `retrieval_config.py`, pure `evaluate()` decision function. Daily limits are checked BEFORE the rate limiter so refusals never consume a slot. Usage is recorded in a `finally` so failed model calls still count.
- `src/usage_tracker.py` gained a **per-request accumulator** (contextvar) so each request's tokens are attributed correctly under concurrency (diffing global totals would not be).
- `src/chat_store.py`: every query scoped by `user_id`; a foreign session id is indistinguishable from a missing one (`SessionNotFound` -> 404).
- `src/memory.py`: history window (6 msgs), `format_history`, rolling summary (`should_summarize` when 6+ new messages age out; one LLM call via the Langfuse-managed `conversation-summary` prompt), long-term notes via `/remember`, `/memories`, `/forget <n|all>` — explicit consent only, PII-redacted, 500-char cap, 50 per user, prompt-injection-looking text refused, recalled by pgvector cosine distance <= 0.5 (top 3), scoped by user.
- `src/chatbot.py`: `handle_message()` orchestrates a turn: validate -> memory command (no quota, no model) -> quota -> session (ownership check) -> redact -> history + recalled notes -> graph (under a usage scope) -> record usage -> persist redacted user message + assistant message with sources/images/flags -> auto title -> best-effort summary.
- `src/api.py` rewritten: `/auth/login|logout|register`, `/me`, `/chat`, `/chat/sessions[/id]`, `/memories[/id]`, `/admin/users` (list/create/patch), `/stats` (admin), `/health`. 401 unauthenticated, 403 wrong role, 404 foreign session, 429 quota (Retry-After for rate limits), 422 bad input. Registration off unless `ALLOW_REGISTRATION=1` and never grants admin; admins cannot deactivate/demote themselves.
- `src/chat_app.py` (Streamlit; run `streamlit run src/chat_app.py`): login/registration, chat, sources + guardrail notes + images, session sidebar, quota meter, notes management. Calls the service layer directly (one process — chosen over a separate API process because the machine is slow); the API is for programmatic use.
- `src/manage_users.py`: admin CLI (`setup-db`, `create-user`, `list-users`, `set-limits`, `deactivate`, `activate`, `reset-password`).
- Pipeline changes: Agent 1 (`agent_transform_route`) now accepts `history` and `notes` and its instruction tells it to resolve follow-ups into self-contained sub-queries (**no extra LLM call**); `retrieval_graph` state gained `history`/`notes`.

**Key design decisions**
- **Follow-ups resolved inside Agent 1**, not by a separate "condense question" call: with Gemini's free tier at 20 calls/day and ~3-4 calls per message, an extra call per turn was not affordable.
- **Cache safety**: the shared query cache is keyed by raw query text. A follow-up ("what about its accuracy?") is meaningless without its conversation, and user notes can shape routing, so when `history` or `notes` is present the raw-query cache is neither read nor written. Cache #2 (keyed on the resolved sub-queries) still applies. Without this, one user's follow-up could be served as another user's answer.
- **Server-side session tokens instead of JWT**: instantly revocable (logout, deactivation, password reset) and needs no JWT library.
- **Long-term memory is explicit-only** (`/remember`): no automatic extraction, which avoids both a per-turn LLM call and consent/PII questions about inferring facts about users.
- **Traces and storage hold redacted text only** (consistent with section 14).

**Changed prompt => resync needed**: `SYSTEM_INSTRUCTION` in `agent_transform_route.py` was extended and a new `conversation-summary` prompt was added. Langfuse still has the OLD `transform-route` version (if it was ever synced) until `python src/prompt_registry.py` is run, and the model would then not see the follow-up rule.

**Decision 2 (user): open sign-up, tight per-user limits.** Anyone may create an account ("They can use, but with limits"): each regular user gets **2 messages per day — a first question and one follow-up** ("I will change that later if needed"; `USER_DAILY_REQUEST_LIMIT`, or per user via `manage_users.py set-limits`). Registration is now ON by default (`ALLOW_REGISTRATION=0` turns it off). Memory commands (`/remember` etc.) do not count as messages. Consequence I flagged and handled: with open sign-up, one person can make many accounts to dodge a per-user limit, so registration is also throttled (`REGISTRATIONS_PER_HOUR`, default 10, service-wide — no per-IP view) and the **global daily cap (Decision 1, default 50)** is the real backstop for the model quota. Not solved: determined abuse (many accounts over many hours, disposable identities) — that would need email verification/CAPTCHA/per-IP limits.

**Known gaps / not built**: password strength rules beyond length 8; email verification / password-reset flows; per-IP rate limiting (limits are per user); lockout and rate limiter are in-memory (reset on restart, not shared across processes — would need Redis for multi-process); no CSRF/HTTPS handling (put behind a TLS proxy for real deployment); Streamlit keeps the token in `st.session_state` (per browser tab session, not persisted across full reloads); image display assumes files are found under `data/images/`; `/remember` embeds with the local bge model, which loads the model on first use (slow on this machine).

**Tests written, not run**: `test_auth`, `test_quotas`, `test_chat_store`, `test_memory`, `test_chatbot`, `test_api` (rewritten), plus additions to `test_usage_tracker`, `test_run_query`, `test_retrieval_graph`, `test_agent_transform_route`, `test_prompt_registry`; helper `tests/fake_db.py`. Database access is exercised only through a fake connection — **no test touches a real Postgres**, so the SQL itself (ON CONFLICT upsert, `<=>` recall, cascades, `make_interval`) is unverified until `setup_app_db` is run and a real integration pass is done.

## 12c. LLM connection and provider fallback

**Why**: user (asked one-by-one, item 1 of "left to build"): the app was Gemini-only, and Gemini's free tier (20 calls/day) makes a single provider a dead end. Requested design, in the user's words: "a separate llm connection file in which multiple llm config settings would be stored, only the API key taken from .env which I will add myself; add 3 — Gemini, Anthropic and OpenAI; then add fallbacks; the whole application can use just the llm connection, and the rest of the fallback can be handled in that separate file."

**Decisions (user)**: fallback order **Gemini, then Anthropic, then OpenAI**; Anthropic model **`claude-haiku-4-5-20251001`**; OpenAI model **`gpt-4o-mini`** (I could not verify current OpenAI model names — one line to change). Settings live in `src/llm_connection.py`; API keys only in `.env` (`GEMINI_API_KEY` exists; user adds `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` themselves — placeholders in `.env.example`).

**Built (written, NOT run)**: `src/llm_connection.py` — `generate(system_instruction, user_content, client=None, order=None) -> LLMResult` (`.text`, `.provider`, `.model`, token counts, `.attempts`, `.fallback_used`). Providers tried in `FALLBACK_ORDER`; skipped when disabled, missing key, SDK not installed, or cooling down; failures are classified (`quota` 429/rate-limit, `auth` 401/403, `unavailable` 5xx/timeout/connection, `other`) and quota/auth/unavailable put the provider on a cooldown (900s/3600s/60s) so a dead provider is not retried on every call; an empty/blocked response falls through without a cooldown; raises `AllProvidersFailed` (with every attempt) if nothing works. Gemini keeps `gemini_retry` (503 backoff + opportunistic prompt caching); Anthropic/OpenAI adapters use their SDKs lazily and record tokens via the new provider-neutral `usage_tracker.record_tokens`. No API key is ever stored in the settings (tested).
- **Callers switched to it**: `agent_transform_route`, `agent_quality_generate` (3 calls), `memory.summarize`, `deepeval_gemini_model` (class name kept; now provider-agnostic). Their `client=` parameter is kept only as a Gemini-client test seam.
- **Also**: `experiment_tracking.collect_versions` logs the LLM order + models with every eval run; admin `/stats` returns `llm_providers` (readiness per provider, no network calls); `tests/conftest.py` has an autouse fixture that blocks the Anthropic/OpenAI adapters in unit tests so real (billed) keys in `.env` can never be used by a test run.
- **Deliberately NOT routed through it (documented exceptions)**: (1) `extract_entities.py` — offline ingestion with its own verification loop, already run; (2) RAGAS judge in `run_evaluation.score_ragas` — RAGAS needs a native provider client via `llm_factory`, not a text-in/text-out function. Left as a follow-up decision.
- **Caveats**: prompt caching only exists for Gemini (other providers get the instruction as a normal system prompt, no caching). Answer quality/format can differ per provider — the agents' JSON-only instructions (routing, quality check) were tuned on Gemini and are untested on Claude/GPT; `parse_response` already tolerates markdown fences. The cooldown state is in memory (per process). `anthropic` is in `requirements.txt` but NOT installed yet; `openai` (1.109.1) is already installed via deepeval.
- **Tests written, not run**: `tests/test_llm_connection.py` (fallback order, cooldowns, key/SDK skipping, error classification, all three adapters against fake SDKs, no double-counting of Gemini usage).

### 12c-2. Model tiers (SUPERSEDES the single-order Gemini -> Anthropic -> OpenAI design above; built, NOT run)
**How it was decided (all by the user, one question at a time)**: the user wanted, in their words, **3 tiers of model files: (1) simple tasks and queries — cheap and fast models; (2) complex queries and reasoning — smarter models; (3) evaluations like DeepEval and RAGAS — with at least 2 models per tier (a fallback), every task simply given a tier (RAGAS through tier 3), and the connection using that tier's model or falling back automatically.** Choices made: the answer-writing tier is **dynamic** (the planner labels each sub-question simple/complex at no extra LLM call); **a tier never spills over into another tier** (a double failure raises `AllProvidersFailed`). Practical problem raised by the user: funding credit on several vendors' accounts. Options weighed (router account / one paid vendor / cloud platform bill / separate accounts); the user first picked OpenAI as the single paid vendor, then paused to check their Gemini credit and finally decided **Google models only, no other providers, may switch to OpenAI or Anthropic later**. Model names were read from the user's own key (`client.models.list()`), not guessed, and the user chose the **"flash-only, free-tier friendly"** set.

**Settings (in `src/llm_connection.py`, the single place to edit)**: `TIERS` — fast: `gemini-3.6-flash` then `gemini-3.5-flash-lite`; reasoning: `gemini-3.8-flash` then `gemini-3.7-flash`; evaluation: `gemini-3.7-flash` then `gemini-3.5-flash`. `TASK_TIERS` — query_planner, quality_check, general_knowledge, conversation_summary, answer_simple -> fast; answer_complex -> reasoning; deepeval_judge, ragas_judge -> evaluation. `PROVIDERS` holds only connection details (the NAME of each key's env var, timeouts); no key is ever stored. `validate_settings()` runs at import: every tier needs >= 2 distinct models from known providers and every task must map to a real tier.

**Behaviour**: `generate(system, user, tier=... | task=...)` walks the tier's models in order; a model is skipped when its provider has no key/SDK or it is cooling down; a failing model is classified (quota / auth / unavailable / other) and put on a per-MODEL cooldown (900s / 3600s / 60s / none) so it isn't retried on every call; an empty/blocked answer falls through without a cooldown; `LLMResult` carries `tier`, `provider`, `model`, tokens and `attempts`. Because cooldowns are per model and **`gemini-3.7-flash` is listed in both the reasoning tier (fallback) and the evaluation tier (primary), it shares one cooldown and one daily quota across both roles.** Quotas are per model, so a different fallback model gets its own allowance. `ready_models(tier)` and `record_failure(...)` expose the same logic to callers that need a native client.

**Call sites**: Agent 1 (`task="query_planner"`, now also returns `complexity`), Agent 2 quality check / general-knowledge / answer writer (`answer_simple` or `answer_complex` from the sub-question's complexity; missing or unrecognised label = complex), `memory.summarize` (`conversation_summary`), DeepEval judge (`deepeval_judge`). **RAGAS** cannot use `generate()` because RAGAS needs a provider's native client: `run_evaluation.score_ragas` builds one from the first ready evaluation-tier model and, if it fails, records the failure (same cooldown) and continues with the next — the least seamless and least certain part (RAGAS's Gemini path uses `instructor`, and the installed version is old; unverified). **Left as is by the user's approval**: `extract_entities.py` (offline ingestion, already run) still calls Gemini directly.

**Also**: admin `/stats` reports `llm_providers` as per-tier, per-model readiness; `experiment_tracking.collect_versions` logs `llm_tier_<name>` = `provider:model,...` with every eval run; the unit-test fixture in `conftest.py` still blocks the unused Anthropic/OpenAI adapters. Anthropic/OpenAI adapters and provider entries remain in the code (unused) so switching a tier later is a `TIERS` edit plus a key in `.env`; the `anthropic` 1.7.0 package installed earlier is currently unused.

**Unknowns / risks**: (1) per-model free quotas and prices are unknown — the user was checking their Gemini credit, and only `gemini-3.6-flash` has ever actually been called, so any of the other five models may be unavailable, rate-limited harder, or behave differently (e.g. JSON-only routing/quality-check prompts were tuned on 3.6-flash); (2) with one vendor the judge and writers are all Gemini-family (some self-grading bias; mitigated only by using different model versions); (3) the planner prompt changed (added `complexity`), so a Langfuse-managed `transform-route` prompt (if ever synced) would be stale; (4) nothing here has been run — expect first-run bugs. **Tests written, not run**: rewritten `tests/test_llm_connection.py` and additions to `test_agent_quality_generate`, `test_agent_transform_route`, `test_retrieval_graph`, `test_deepeval_gemini_model`, `test_memory`.

## 13. Testing

pytest, one file per module, written alongside each module. Policy: run the full suite only when shared code changed; isolated new files run just their own tests. **Current instruction from the user: write tests but do not run them until told.** Written and unrun: `test_run_evaluation`, `test_prompt_registry`, `test_experiment_tracking`, `test_langfuse_client`, `test_usage_tracker`, `test_api`, `test_deepeval_gemini_model` (ran, passed), `test_generate_golden_set` (ran, passed). Also unrun since editing: the agent/graph/gemini_retry tests (touched by the prompt-registry and usage-tracker wiring; `gemini_retry` is shared code, so those should be re-run first when testing resumes).

**Test isolation (added 2026-09-20).** `tests/conftest.py` has an autouse guard: unit tests cannot reach the real Postgres (`psycopg.connect`), Neo4j (`GraphDatabase.driver`) or Gemini (`genai.Client`), and the Anthropic/OpenAI adapters are blocked; a test that forgets to fake one fails with a clear message instead of silently spending quota. Caveat: the guard may expose existing tests that quietly relied on a real service. Faker (`Faker==40.39.0`, fixed seed 20260920, helpers in `tests/factories.py`) supplies realistic data VALUES; behaviour of services is still faked with stand-ins. `tests/test_faker_properties.py` holds property-style tests (random emails, usernames, passwords, quota decisions). **Integration tests** (`tests/integration/`, marker `integration`) run the real application SQL against the real Postgres in a throwaway schema `it_<hex>` that is dropped afterwards; skipped unless `--run-integration` or `RUN_INTEGRATION=1`: `python -m pytest tests/integration --run-integration`. Written, not run. Model calls and the embedding model stay faked there.

**Test layout (2026-09-20, verified: 418 passed, 25 skipped by default).** Folders describe what a test NEEDS; the future CI/CD pipeline decides WHEN each runs. `tests/unit/` (all fakes; every push/PR), `tests/integration/` (real Postgres, throwaway schema; opt-in `--run-integration`/`RUN_INTEGRATION=1`; cheap, so also fine on push if CI has a pgvector Postgres service), `tests/live/` (real Gemini + Neo4j + full pipeline, spends quota; opt-in `--run-live`/`RUN_LIVE=1`; for a manual button or weekly schedule; 3 smoke tests written, never run). The marker for a folder is applied automatically by `tests/conftest.py` (no per-file marker needed). Shared helpers stay in `tests/` (`conftest.py`, `factories.py`, `fake_db.py`). Suggested CI mapping: push/PR = unit (+integration); nightly = all non-live + dependency-drift check; weekly/manual = live + eval run on the golden set. No CI file exists yet (user: pipeline will be created later).

**Audit fixes (2026-09-20, written, not run).** Quota is charged only for real answers (failed runs and messages refused before any model call are free; new sessions are created only after success). `chatbot.ServiceUnavailable` turns "all models in a tier failed" and "Neo4j unreachable" into a friendly warning (UI) / 503 with `Retry-After` (API). Evaluation runs pass `use_cache=False` so scores measure the pipeline, not the cache. Injection guardrail matches commands aimed at the assistant, not mentions of attack topics (the corpus contains an AI-security paper).

**Capacity limits (2026-09-20, written, not run).** `DAILY_ACTIVE_USER_CAP` (default 20): max DIFFERENT users served per day; a user already served today is never cut off, only a new user is refused when full; admins exempt; refusal reason `daily_user_cap`. `MAX_CONCURRENT_REQUESTS` (default 3): questions processed at the same moment, enforced by an in-memory non-blocking `ConcurrencyLimiter` around the graph run in `chatbot.handle_message`; when full the user gets reason `busy` (429 + `Retry-After: 10` in the API) and is not charged. Caveat: the concurrency gate is per process, so the Streamlit app and the API each have their own 3 slots; the daily user cap counts users in the database, so it is shared. Defaults CONFIRMED by the user (2026-09-20): with 2 messages per user, 20 users = 40 requests, under the global cap of 50 (which itself is optimistic for Gemini's free tier).

**Online evaluation (2026-09-20, built; unit + integration tests pass; never exercised with a real model).** The "in production" half of evaluation (offline = golden set before release; online = live traffic after). (1) **Feedback**: `src/feedback.py`, thumbs up/down under every answer in the Streamlit UI and `POST /chat/messages/{id}/feedback`; one rating per user per answer (upsert), owner-only, optional comment PII-redacted and capped at 500 chars; table `answer_feedback`. (2) **Sampled judge**: `src/online_eval.py`, `ONLINE_EVAL_SAMPLE_RATE` (default 0.1, 0 = off) of real answers are scored 0-1 for faithfulness (against the retrieved context; null when there is no context) and relevance by an evaluation-tier model (`task="online_judge"`) in a daemon thread AFTER the reply is stored: never delays or charges the user, never fails a turn; blocked/empty answers are skipped; question/context/answer are fenced as untrusted data in the judge prompt; malformed or out-of-range replies are dropped, not stored; table `online_eval_scores`. (3) **Health data**: every stored answer's metadata now carries `latency_s`, `prompt_tokens`, `output_tokens` (plus the existing cache_hit/blocked). (4) **Report**: `src/online_report.py` (`python src/online_report.py [days]`, admin sidebar panel in the UI, `GET /admin/online-metrics?days=`): `summary` (answers, avg/p95 latency, cache-hit and block rates, tokens, thumbs, satisfaction, judged count, avg faithfulness/relevance), `alerts` (each with a minimum sample size so a few answers cannot cause a false alarm: satisfaction < 60% with >= 10 votes, faithfulness/relevance < 0.7 with >= 5 judged, p95 > 60 s or block rate > 20% with >= 10 answers), and `review_queue` (answers judged < 0.5 on either score OR thumbs-down, with the redacted question) = the feedback loop into the golden set. Caveats: sampling shares the evaluation-tier free quota with offline runs; the judge is an LLM and can be wrong or biased (treat scores as a trend, not truth; the judge model is stored per score); no drift/topic monitoring, no automatic golden-set promotion (a human writes the reference answer); the two new tables were created on the real database with `python src/setup_app_db.py` (idempotent) on 2026-09-20.

**Status checkpoint (2026-09-20).** Test suite: 565 tests (529 unit, 33 integration, 3 live), all non-live passing. Verified live: query pipeline, chat handler, model failover, guardrails (attacks blocked at input with 0 model calls), follow-up rewrite and memory commands. Cost/latency fixes measured on one question: 5 -> 3 model calls, 32,015 -> 14,609 prompt tokens, 86.9 -> 58.2 s (n=1). Guardrail fixes after live probing: output-guardrail leak pattern made first-person only; clinical-advice framing now raises `medical_advice_framing` and a fixed non-LLM safety note (`retrieval_graph.MEDICAL_NOTE`) is added on every answer path incl. cache hits, after caching; citation regex accepts `p. 15`, `p. 1, 8`, `pp. 3-5`; long-term note recall prefixes the user's previous question (cosine distance 0.576 -> 0.207). Open: recall threshold 0.5 too loose (on-topic 0.18-0.27, off-topic 0.42-0.58 on 6 samples; 0.35 would separate); citation check verifies page retrieval, not the claim; planner over-labels `complex`; two papers lack graph entity links; login+chat over HTTP, the browser UI, the online judge, a full offline eval, and Langfuse are unproven or off. See `context_prompt.md` (CURRENT STATE block) and `improvement_stories.md`.

## 14. Observability

- Logs/traces must store redacted queries only, never raw PII. **Bug caught while wiring tracing**: the Langfuse callback records the graph's initial input, which contained the raw (unredacted) query. Fixed: `run_query.initial_state()` now redacts PII before the query enters the graph (redaction is idempotent, so the graph's own input-guardrail node redacting again is harmless). Covered by `tests/test_run_query.py` (written, not run). Remaining gap: the API layer does not log request bodies, so nothing else stores the raw query — keep it that way.
- Stack: self-hosted Langfuse (see Tracing).

## 15. Tracing

Langfuse, self-hosted via Docker compose in `langfuse/` (postgres, clickhouse, redis, minio, web, worker; host Postgres port remapped to 5433 because the project's own Postgres owns 5432). Chosen over LangSmith: free/self-hostable vs cloud-only quota. LangGraph runs are traced through the callback handler in `langfuse_client.get_callbacks()`.

## 16. Cost Tracking

`src/usage_tracker.py`: thread-safe running totals of Gemini calls, prompt/output/cached tokens, recorded inside `gemini_retry` after every successful call. Estimated cost = tokens x `GEMINI_PRICE_PER_M_INPUT` / `_OUTPUT` env vars (default 0, the free tier). Tokens are measured; dollar figures are only as good as the configured prices. Surfaced in `GET /stats` and as `usage_*` metrics on each MLflow eval run. Not tracked: embedding-model compute (local), Postgres/Neo4j/Docker costs.

## 17. Latency Monitoring

Per-request latency in the API response and an `X-Latency-Seconds` header; `GET /stats` reports avg/p50/p95/max over the last 1000 requests (in-memory, resets on restart). Eval runs record `latency_s` per example. Per-stage latency comes from Langfuse traces.

## 18. Deployment

`Dockerfile` for the API (python:3.11-slim, installs `requirements.txt`, runs `uvicorn api:app`); `.dockerignore` excludes venv, data, tests, docs and secrets. **Written, not built or run.** Postgres, Neo4j and Gemini are external services configured by environment variables. Model weights download on first use (mount a volume for the HuggingFace cache). Caveats: the image will be large (torch, transformers, docling); inside a container `DATABASE_URL`/`NEO4J_URI` must point at the host or other containers, not `localhost`.
