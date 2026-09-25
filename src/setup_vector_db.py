from db import get_connection, SCHEMA_NAME

TEXT_EMBEDDING_DIM = 768
IMAGE_EMBEDDING_DIM = 512

SCHEMA_SQL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME};

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.text_parents (
    parent_id TEXT PRIMARY KEY,
    source_pdf TEXT NOT NULL,
    section TEXT,
    page_start INTEGER,
    page_end INTEGER,
    domain TEXT NOT NULL,
    text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.text_chunks (
    chunk_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL REFERENCES {SCHEMA_NAME}.text_parents(parent_id),
    source_pdf TEXT NOT NULL,
    section TEXT,
    page_start INTEGER,
    page_end INTEGER,
    domain TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    embedding vector({TEXT_EMBEDDING_DIM}) NOT NULL,
    text_search tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS text_chunks_embedding_idx
    ON {SCHEMA_NAME}.text_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS text_chunks_domain_idx
    ON {SCHEMA_NAME}.text_chunks (domain);

CREATE INDEX IF NOT EXISTS text_chunks_text_search_idx
    ON {SCHEMA_NAME}.text_chunks USING gin (text_search);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.images (
    image_file TEXT PRIMARY KEY,
    source_pdf TEXT NOT NULL,
    page INTEGER,
    domain TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    embedding vector({IMAGE_EMBEDDING_DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS images_embedding_idx
    ON {SCHEMA_NAME}.images USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS images_domain_idx
    ON {SCHEMA_NAME}.images (domain);

-- One row per (document, content version). A content change soft-deletes the
-- old active row (status='deleted', deleted_at set) and inserts a new active
-- one, instead of overwriting - keeps prior versions around for rollback.
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.ingestion_manifest (
    source_pdf TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deleted')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    PRIMARY KEY (source_pdf, content_hash)
);

CREATE INDEX IF NOT EXISTS ingestion_manifest_status_idx
    ON {SCHEMA_NAME}.ingestion_manifest (status);

-- One row per orchestrator run (manual or scheduled).
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.ingestion_runs (
    run_id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'succeeded', 'failed')),
    trigger TEXT NOT NULL DEFAULT 'manual' CHECK (trigger IN ('manual', 'scheduled'))
);

-- One row per stage within a run. Doubles as human-review detail and as
-- the state the orchestrator reads to auto-resume an unfinished run.
-- prompt_tokens/output_tokens intentionally duplicate usage_tracker (per-run
-- scoped view + a tally cross-check against the global tracker, on purpose).
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.ingestion_run_stages (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES {SCHEMA_NAME}.ingestion_runs(run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    latency_seconds NUMERIC,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS ingestion_run_stages_run_idx
    ON {SCHEMA_NAME}.ingestion_run_stages (run_id);
"""


def setup_vector_db():
    conn = get_connection()
    conn.execute(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print(f"Schema '{SCHEMA_NAME}' and tables (text_parents, text_chunks, images, "
          f"ingestion_manifest, ingestion_runs, ingestion_run_stages) ready.")


if __name__ == "__main__":
    setup_vector_db()
