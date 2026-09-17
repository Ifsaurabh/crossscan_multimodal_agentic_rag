# RAG Project Plan

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

- Multimodal embedding approach needed (e.g. CLIP-style or dual text/image embeddings in a shared or linked space).

## 6. Vector Storage

- Store both text and image embeddings, with metadata linking images back to source PDF/page/nearby text.
- **Namespaces**: partitioned by **content type + domain** (e.g. `text-lung-cancer`, `text-land-cover`, `image-lung-cancer`, `image-land-cover`) — separates text/image (different embedding types/dimensions) and lets the agent narrow search scope by topic.

## 6b. Graph Storage

- Entity/relationship extraction (methods, datasets, techniques, domains) for graph queries alongside vector retrieval.
- **Image-text relationships**: captures which images relate to which paragraph/section (e.g. caption/figure mentions) as an explicit graph edge, since chunking (Stage 4) deliberately stays text-only and doesn't embed image references into chunks.

## 7. Input Guardrails

- PII detection/redaction on incoming queries (e.g. names, emails) before any processing, logging, or LLM call.
- Medical-advice framing checks (corpus includes lung-cancer research; guard against the system being used as a diagnostic/treatment advisor).
- Off-topic / prompt-injection filtering before queries reach the agent.

## 8. Retrieval

- Agent selects vector search, graph query, or both depending on query type.

## 9. Generation

- Vision-capable LLM needed to reason over retrieved images, not just text.
- **Citations: inline, per claim** — each generated statement tagged with its source at point of use (e.g. "[LungPaper.pdf, p.4]"), not just a source list appended at the end. More rigorous, closer to academic citation style; adds complexity to generation prompting (needs chunk metadata — source_pdf, page — threaded through into the prompt/output).

## 10. Output Guardrails

- Grounding/hallucination checks (claims must trace to retrieved text/images, not be invented).
- Avoid overstating research findings as clinical fact.

## 11. Evaluation

## 12. API Integration

## 13. Testing

## 14. Observability

- Logs/traces must store redacted queries only, never raw PII.

## 15. Tracing

## 16. Cost Tracking

## 17. Latency Monitoring

## 18. Deployment
