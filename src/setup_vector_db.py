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
    embedding vector({TEXT_EMBEDDING_DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS text_chunks_embedding_idx
    ON {SCHEMA_NAME}.text_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS text_chunks_domain_idx
    ON {SCHEMA_NAME}.text_chunks (domain);

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
"""


def setup_vector_db():
    conn = get_connection()
    conn.execute(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print(f"Schema '{SCHEMA_NAME}' and tables (text_parents, text_chunks, images) ready.")


if __name__ == "__main__":
    setup_vector_db()
