import hashlib
import json

from shared.db import get_connection, SCHEMA_NAME


def setup_sql(schema: str) -> str:
    """The SQL that creates the answer cache table in `schema`."""
    return f"""
CREATE TABLE IF NOT EXISTS {schema}.query_cache (
    query_hash TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    chunks_retrieved JSONB,
    answer TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    sources TEXT[]
);

-- The documents (`<domain>/<file>`) an entry was built from, so the entries that cite a document can be found fast
-- (a deleted document takes them with it). ADD COLUMN IF NOT EXISTS: a table made before this column gets it.
ALTER TABLE {schema}.query_cache ADD COLUMN IF NOT EXISTS sources TEXT[];

CREATE INDEX IF NOT EXISTS query_cache_sources_idx
    ON {schema}.query_cache USING gin (sources);
"""


SETUP_SQL = setup_sql(SCHEMA_NAME)


def setup_query_cache():
    conn = get_connection()
    conn.execute(SETUP_SQL)
    conn.commit()
    conn.close()
    print(f"query_cache table ready in schema '{SCHEMA_NAME}'.")


def hash_query(query_text: str) -> str:
    normalized = query_text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_cached(conn, query_text: str):
    query_hash = hash_query(query_text)
    cur = conn.execute(
        f"SELECT chunks_retrieved, answer FROM {SCHEMA_NAME}.query_cache WHERE query_hash = %s",
        (query_hash,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    chunks_retrieved, answer = row
    return {"chunks_retrieved": chunks_retrieved, "answer": answer}


def sources_of(chunks_retrieved) -> list:
    """The documents an answer was built from: the distinct `source_pdf` of its chunks, sorted."""
    return sorted({c["source_pdf"] for c in (chunks_retrieved or []) if isinstance(c, dict) and c.get("source_pdf")})


def write_cache(conn, query_text: str, chunks_retrieved, answer: str):
    query_hash = hash_query(query_text)
    conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.query_cache (query_hash, query_text, chunks_retrieved, answer, sources)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (query_hash) DO UPDATE SET
                chunks_retrieved = EXCLUDED.chunks_retrieved, answer = EXCLUDED.answer, sources = EXCLUDED.sources""",
        (query_hash, query_text, json.dumps(chunks_retrieved), answer, sources_of(chunks_retrieved)),
    )
    conn.commit()


if __name__ == "__main__":
    setup_query_cache()
