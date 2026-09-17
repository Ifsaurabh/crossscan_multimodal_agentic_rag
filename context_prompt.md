# RAG-NEW Project Context Prompt

_Last updated: 2026-09-17 20:25 (auto-checkpoint, hourly while session is active)_

You are helping build a demonstration RAG project in the `RAG-NEW` folder.

## Project Scope

Work only inside `RAG-NEW`. Ignore all other folders in the workspace unless explicitly requested.

This is an **agentic, multimodal RAG project** with a **graph DB alongside the vector DB**, built over a dataset of 12 research-paper PDFs (lung cancer / medical imaging, and land cover / remote sensing, plus a few unclassified).

- **Multimodal**: text + images (figures, charts, CT/satellite imagery) extracted, embedded, and retrievable together.
- **Graph DB**: alongside vector DB, to capture explicit relationships (shared methods, datasets, techniques across papers, and image-to-text links) that similarity search misses. Hybrid retrieval, not a replacement for vector search.
- **Agentic**: orchestrator/agent decides which retrieval tool to use (vector vs. graph vs. both) and can decompose multi-step questions rather than a single fixed retrieve-then-generate flow.
- **Use case anchor**: a research-assistant tool for literature review — find relevant findings across papers, surface figures/charts in context, and identify shared methodology across domains (e.g. CNN used in both medical and remote-sensing papers).
- **PII scope**: not about the documents (published papers, no patient/personal data) — about user **queries** that may contain PII (e.g. names, emails). Redact at the input boundary before logging/tracing or sending to any LLM API.
- **Reproducibility**: dataset is downloaded via `kagglehub` (not stored as a binary in the repo); all data under `data/` (except small JSON reports) is regenerable from code and is gitignored.

## Collaboration Style

This is a learning and demonstration project. The user wants to understand, discuss, and build the system as they go.

- Do not create files or write code automatically unless the user explicitly asks.
- Explain the purpose of each step before implementation; ask one question at a time when scoping a decision.
- Prefer small, incremental changes.
- Discuss design choices and tradeoffs in accessible technical language.
- Run checks after changes when appropriate.
- Keep the user involved in decisions rather than completing the whole project at once.
- Don't commit to git unless explicitly asked, even when a repo exists.

## Current Status

- **Stage 1 (Data Loading): done.** Dataset downloaded via `kagglehub` (`saibhossain/rag-practice`, 12 PDFs, no credentials needed) and copied to `data/raw/` via `src/load_data.py`. `archive.zip` removed — no longer needed.
- **Stage 2 (Data Cleaning / extraction): done, reworked for page tracking.**
  - Text extraction: **Docling**, using `document.iterate_items()` (label, level, page_no, text) instead of markdown+regex — more robust, gives real page numbers. Output: `data/text/*.json` (blocks) via `src/extract_text.py`.
  - Image extraction: **PyMuPDF**. Output: images + `data/images/metadata.json` via `src/extract_images.py`.
  - **Quality check** (`src/quality_check.py`): automated heuristic checks (chars-per-page, headings, garbage-char ratio for text; tiny/corrupt images). Finds AND drops flagged images in the same run. Latest clean run: 0/12 text flagged, 0/174 images flagged.
- **Stage 3 (Document Preparation): done, reworked for page ranges.** `src/prepare_documents.py` — groups blocks into sections by `section_header` items, computes `page_start`/`page_end` per section from block-level page numbers, strips references/bibliography sections. Output: `data/prepared/*.json`. All 12/12 references sections correctly dropped.
- **Stage 3b (Ingestion Guardrails): done, text AND images.**
  - Text (`src/ingestion_guardrails.py`): email PII → redacted (~28 total across corpus). Phone-number regex was tried and removed — false-matched journal citation numbers. Unsafe-content keyword filter: 0 hits (expected). Output: `data/guarded/*.json` + `data/guardrail_report.json`.
  - Images (`src/image_guardrails.py`): free/local vision model **Hugging Face `Falconsai/nsfw_image_detection`** (no API cost), batch-classifies all extracted images, drops flagged ones (score ≥0.5) from disk + `metadata.json`. Result: 0/174 flagged (expected — research figures, not photographic content). Output: `data/image_guardrail_report.json`.
- **Chunking (Stage 4) design fully scoped, NOT YET IMPLEMENTED:**
  - Strategy: **parent-child chunking**. Parent = section (~1500-2000 token cap, split into sub-chunks if oversized). Child = paragraph (~300-500 token cap, recursively split via sentences/windows if oversized). 20% overlap both levels.
  - Metadata (core set): `source_pdf`, `section`, `chunk_id`, `parent_id` (children), `chunk_type`, `page_start`/`page_end`, `domain`.
  - Tokenizer: `tiktoken`.
  - **Text-only** — chunks do not carry image references; image-to-text relationships deferred to Stage 6b (Graph Storage) as explicit graph edges instead.
  - Needs a `chunking_version`/config tag in metadata + report (config-versioning decision, not yet implemented).
- **Other design decisions recorded (not yet built)**: Vector Storage namespaces = content-type + domain (e.g. `text-lung-cancer`, `image-land-cover`); Generation citations = inline per-claim (not end-of-answer list).
- **Infra housekeeping done this session:**
  - Isolated project venv, renamed from `venv311` to **`venv-rag-new`** (to avoid confusion with the shared `G:\Projects\venv311`) — direct invocation via `venv-rag-new/Scripts/python.exe` works fine after rename (no activation used).
  - `requirements.txt` added (kagglehub, pymupdf, docling, pillow, transformers, torch, tiktoken).
  - **Git initialized** (local only, not yet committed — user wants to commit once chunking code + config-versioning are ready). `.gitignore` excludes `venv-rag-new/`, `data/raw|text|images|prepared|guarded|chunks/`, `.env`; tracks code, `plannings.md`, `context_prompt.md`, `.env.example`, `requirements.txt`, and the small `*_report.json` files.
  - Private GitHub remote requested by user but not yet set up — `gh` CLI is not installed in this environment, so it couldn't be created from here.
- Dataset: mix of lung-cancer/medical-imaging papers, land-cover/remote-sensing papers, and a few unclassified arXiv-style files.
- Full stage-by-stage plan and scope notes tracked in `plannings.md` (more detailed/authoritative than this file for exact design rationale).

## Current Files

- `.env.example`, `requirements.txt`, `.gitignore` — project root.
- `plannings.md` — stage plan + scope decisions (detailed).
- `src/load_data.py` — Stage 1 kagglehub download.
- `src/extract_text.py` — Stage 2 Docling text+page extraction (block-based).
- `src/extract_images.py` — Stage 2 PyMuPDF image extraction.
- `src/quality_check.py` — Stage 2 automated quality gate.
- `src/prepare_documents.py` — Stage 3 section-splitting + page ranges + references stripping.
- `src/ingestion_guardrails.py` — Stage 3b text PII redaction + unsafe-content filtering.
- `src/image_guardrails.py` — Stage 3b image NSFW/content-safety filtering.
- `data/raw/`, `data/text/`, `data/images/`, `data/prepared/`, `data/guarded/` — pipeline outputs (all gitignored, regenerable).
- `data/quality_report.json`, `data/guardrail_report.json`, `data/image_guardrail_report.json` — small tracked reports.
- `venv-rag-new/` — project-local virtual environment (Python 3.11).
- `.git/` — initialized, nothing committed yet.

## Immediate Next Direction

Build the Stage 4 chunking script (parent-child, tiktoken, config-versioned) per the fully-scoped design above. Once chunking code + config-versioning are in place, user wants to make the first git commit. After that: Stage 5 (Embedding), Stage 6 (Vector Storage with namespaces), Stage 6b (Graph Storage, including image-text relationships).
