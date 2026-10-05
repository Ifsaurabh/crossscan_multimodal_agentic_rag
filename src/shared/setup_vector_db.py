from shared.db import get_connection, SCHEMA_NAME

TEXT_EMBEDDING_DIM = 768
IMAGE_EMBEDDING_DIM = 512

def schema_sql(schema: str) -> str:
    """The SQL that creates the schema and the vector tables in `schema`."""
    return f"""
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.text_parents (
    parent_id TEXT PRIMARY KEY,
    source_pdf TEXT NOT NULL,
    section TEXT,
    page_start INTEGER,
    page_end INTEGER,
    domain TEXT NOT NULL,
    text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS {schema}.text_chunks (
    chunk_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL REFERENCES {schema}.text_parents(parent_id),
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
    ON {schema}.text_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS text_chunks_domain_idx
    ON {schema}.text_chunks (domain);

CREATE INDEX IF NOT EXISTS text_chunks_text_search_idx
    ON {schema}.text_chunks USING gin (text_search);

CREATE TABLE IF NOT EXISTS {schema}.images (
    image_file TEXT PRIMARY KEY,
    source_pdf TEXT NOT NULL,
    page INTEGER,
    parent_id TEXT,
    domain TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    embedding vector({IMAGE_EMBEDDING_DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS images_embedding_idx
    ON {schema}.images USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS images_domain_idx
    ON {schema}.images (domain);

-- The parent chunk an image belongs to (the first parent whose pages hold the image's page), set at ingestion.
-- ADD COLUMN IF NOT EXISTS: a schema created before this column existed gets it, nothing else changes.
ALTER TABLE {schema}.images ADD COLUMN IF NOT EXISTS parent_id TEXT;

CREATE INDEX IF NOT EXISTS images_parent_idx
    ON {schema}.images (parent_id);

-- One row per (document, content version). A content change soft-deletes the
-- old active row (status='deleted', deleted_at set) and inserts a new active
-- one, instead of overwriting - keeps prior versions around for rollback.
CREATE TABLE IF NOT EXISTS {schema}.ingestion_manifest (
    source_pdf TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deleted')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    PRIMARY KEY (source_pdf, content_hash)
);

CREATE INDEX IF NOT EXISTS ingestion_manifest_status_idx
    ON {schema}.ingestion_manifest (status);
"""


SCHEMA_SQL = schema_sql(SCHEMA_NAME)


def setup_vector_db():
    conn = get_connection()
    conn.execute(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print(f"Schema '{SCHEMA_NAME}' and tables (text_parents, text_chunks, images, ingestion_manifest) ready. "
          "(The records of every ingestion attempt are in ingestion_reports: python -m ingestion.ingestion_reports --setup.)")


if __name__ == "__main__":
    setup_vector_db()
