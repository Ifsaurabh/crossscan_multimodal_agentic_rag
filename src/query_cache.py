import hashlib
import json

from db import get_connection, SCHEMA_NAME

SETUP_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.query_cache (
    query_hash TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    chunks_retrieved JSONB,
    answer TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""


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


def write_cache(conn, query_text: str, chunks_retrieved, answer: str):
    query_hash = hash_query(query_text)
    conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.query_cache (query_hash, query_text, chunks_retrieved, answer)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (query_hash) DO UPDATE SET
                chunks_retrieved = EXCLUDED.chunks_retrieved, answer = EXCLUDED.answer""",
        (query_hash, query_text, json.dumps(chunks_retrieved), answer),
    )
    conn.commit()


if __name__ == "__main__":
    setup_query_cache()
