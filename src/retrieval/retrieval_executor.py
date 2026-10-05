import contextlib
import sys
import threading

from sentence_transformers import SentenceTransformer

from retrieval import tracing
from shared.db import connection, SCHEMA_NAME
from shared.embedding_config import TEXT_MODEL_NAME
from retrieval.retrieval_config import VECTOR_TOP_K, TABLE_TOP_K, MIN_TABLE_SIMILARITY, MAX_PARALLEL_SEARCHES

_model = None
# A background warm-up and a first question can ask for the model at the same
# moment; the lock makes the second caller wait for the load instead of
# starting a duplicate one.
_model_lock = threading.Lock()
# One encode at a time: searches of several sub-questions can run in parallel, and the embedding is a few milliseconds.
_embed_lock = threading.Lock()
# How many database searches run at once in this process (the size of the connection pool).
_search_slots = threading.BoundedSemaphore(MAX_PARALLEL_SEARCHES)


def get_embedding_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = SentenceTransformer(TEXT_MODEL_NAME)
    return _model


def embed_query(query_text: str):
    model = get_embedding_model()
    with _embed_lock, tracing.step("embed_query", timing="embed_s", input=query_text, model=TEXT_MODEL_NAME):
        return model.encode(query_text, normalize_embeddings=True)


def embed_queries(texts: list) -> list:
    """The embeddings of several texts in ONE call (the variants of a sub-question), in the same order."""
    if not texts:
        return []
    model = get_embedding_model()
    with _embed_lock, tracing.step("embed_query", timing="embed_s", variants=len(texts), model=TEXT_MODEL_NAME):
        return list(model.encode(list(texts), normalize_embeddings=True))


@contextlib.contextmanager
def _borrow(name: str, **metadata):
    """A pooled connection, traced: the wait for the pool is its own span (`db_pool_wait`), then the work on the
    connection is a span named `name`. Both count towards `db_s`. The connection is handed back exactly as `with
    connection() as conn` would (an error inside reaches the pool, so it can discard a broken connection)."""
    with tracing.step("db_pool_wait", timing="db_s"):
        _search_slots.acquire()  # at most MAX_PARALLEL_SEARCHES at once, so parallel searches cannot starve the pool
        try:
            borrowed = connection()
            conn = borrowed.__enter__()
        except BaseException:
            _search_slots.release()
            raise
    try:
        with tracing.step(name, timing="db_s", **metadata) as span:
            yield conn, span
    except BaseException:
        borrowed.__exit__(*sys.exc_info())
        raise
    else:
        borrowed.__exit__(None, None, None)
    finally:
        _search_slots.release()


def _query(conn, name: str, sql: str, params):
    """One search, its own span with the number of rows it returned."""
    with tracing.step(name) as span:
        rows = conn.execute(sql, params).fetchall()
        span.update(metadata={"rows": len(rows)})
    return rows


def vector_search(query_text: str, domain: str = None, source_pdfs: list = None, top_k: int = VECTOR_TOP_K,
                  query_vec=None):
    """Simple semantic-only vector search over text_chunks, joined with parent
    text for generation context. Optionally narrowed by domain or a list of
    source_pdf values. `query_vec` is the text's embedding when the caller has already made it (in a batch)."""
    if query_vec is None:
        query_vec = embed_query(query_text)

    where_clauses = []
    where_values = []

    if domain:
        where_clauses.append("c.domain = %s")
        where_values.append(domain)
    if source_pdfs:
        where_clauses.append("c.source_pdf = ANY(%s)")
        where_values.append(source_pdfs)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with _borrow("search_vector", top_k=top_k, filtered=bool(where_clauses)) as (conn, span):
        cur = conn.execute(
            f"""SELECT c.chunk_id, c.text, c.source_pdf, c.section, c.page_start, c.page_end,
                       c.parent_id, p.text AS parent_text, 1 - (c.embedding <=> %s) AS similarity
                FROM {SCHEMA_NAME}.text_chunks c
                JOIN {SCHEMA_NAME}.text_parents p ON c.parent_id = p.parent_id
                {where_sql}
                ORDER BY c.embedding <=> %s
                LIMIT %s""",
            [query_vec] + where_values + [query_vec, top_k],
        )
        rows = cur.fetchall()
        span.update(metadata={"rows": len(rows)})

    return [
        {
            "chunk_id": r[0], "text": r[1], "source_pdf": r[2], "section": r[3],
            "page_start": r[4], "page_end": r[5], "parent_id": r[6],
            "parent_text": r[7], "similarity": float(r[8]),
        }
        for r in rows
    ]


def hybrid_vector_search(query_text: str, domain: str = None, top_k: int = VECTOR_TOP_K, query_vec=None):
    """Semantic (pgvector) + keyword (Postgres full-text) combined via
    reciprocal rank fusion. `query_vec` is the text's embedding when the caller has already made it (in a batch)."""
    if query_vec is None:
        query_vec = embed_query(query_text)
    fetch_k = top_k * 3

    where_domain = "AND c.domain = %s" if domain else ""

    semantic_params = [query_vec]
    if domain:
        semantic_params.append(domain)
    semantic_params += [query_vec, fetch_k]

    with _borrow("search_hybrid", top_k=top_k, fetch_k=fetch_k, filtered=bool(domain)) as (conn, span):
        semantic_rows = _query(
            conn, "search_semantic",
            f"""SELECT c.chunk_id, 1 - (c.embedding <=> %s) AS similarity
                FROM {SCHEMA_NAME}.text_chunks c
                WHERE TRUE {where_domain}
                ORDER BY c.embedding <=> %s
                LIMIT %s""",
            semantic_params,
        )
        semantic_ranked = [row[0] for row in semantic_rows]

        keyword_rows = _query(
            conn, "search_keyword",
            f"""SELECT c.chunk_id
                FROM {SCHEMA_NAME}.text_chunks c
                WHERE c.text_search @@ plainto_tsquery('english', %s) {where_domain}
                ORDER BY ts_rank_cd(c.text_search, plainto_tsquery('english', %s)) DESC
                LIMIT %s""",
            [query_text] + ([domain] if domain else []) + [query_text, fetch_k],
        )
        keyword_ranked = [row[0] for row in keyword_rows]

        rrf_scores = {}
        k = 60
        for rank, chunk_id in enumerate(semantic_ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        for rank, chunk_id in enumerate(keyword_ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        top_chunk_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        span.update(metadata={"semantic_rows": len(semantic_ranked), "keyword_rows": len(keyword_ranked),
                              "fused": len(top_chunk_ids)})
        if not top_chunk_ids:
            return []

        parent_rows = _query(
            conn, "fetch_parents",
            f"""SELECT c.chunk_id, c.text, c.source_pdf, c.section, c.page_start, c.page_end,
                       c.parent_id, p.text AS parent_text
                FROM {SCHEMA_NAME}.text_chunks c
                JOIN {SCHEMA_NAME}.text_parents p ON c.parent_id = p.parent_id
                WHERE c.chunk_id = ANY(%s)""",
            (top_chunk_ids,),
        )
        rows = {r[0]: r for r in parent_rows}

    return [
        {
            "chunk_id": cid, "text": rows[cid][1], "source_pdf": rows[cid][2],
            "section": rows[cid][3], "page_start": rows[cid][4], "page_end": rows[cid][5],
            "parent_id": rows[cid][6], "parent_text": rows[cid][7], "rrf_score": rrf_scores[cid],
        }
        for cid in top_chunk_ids if cid in rows
    ]


def get_images_for_parents(parent_ids: list = None):
    """The pictures of the given parent chunks, from Postgres. An image was linked to its parent at ingestion (the
    first parent whose pages hold the image's page), so this is one lookup by parent id."""
    if not parent_ids:
        return []

    with _borrow("lookup_images", parents=len(parent_ids)) as (conn, span):
        rows = conn.execute(
            f"""SELECT image_file, page, parent_id, source_pdf
                FROM {SCHEMA_NAME}.images
                WHERE parent_id = ANY(%s)
                ORDER BY source_pdf, page, image_file""",
            (list(parent_ids),),
        ).fetchall()
        span.update(metadata={"rows": len(rows)})
    return [{"image_file": r[0], "page": r[1], "parent_id": r[2], "source_pdf": r[3]} for r in rows]


def _table_dict(row, similarity=None):
    table = {"table_id": row[0], "text": row[1], "caption": row[2] or "", "page": row[3], "parent_id": row[4],
             "source_pdf": row[5]}
    if similarity is not None:
        table["similarity"] = float(similarity)
    return table


def get_tables_for_parents(parent_ids: list = None):
    """The tables of the given parent chunks, from Postgres (the table text is stored with the row, so it is returned
    directly). The link to the parent was made at ingestion, by section heading and page."""
    if not parent_ids:
        return []

    with _borrow("lookup_tables", parents=len(parent_ids)) as (conn, span):
        rows = conn.execute(
            f"""SELECT table_id, text, caption, page, parent_id, source_pdf
                FROM {SCHEMA_NAME}.doc_tables
                WHERE parent_id = ANY(%s)
                ORDER BY source_pdf, page, table_id""",
            (list(parent_ids),),
        ).fetchall()
        span.update(metadata={"rows": len(rows)})
    return [_table_dict(r) for r in rows]


def search_tables(query_text: str, top_k: int = TABLE_TOP_K, min_similarity: float = MIN_TABLE_SIMILARITY):
    """The tables whose embedding (caption + table text) is closest to the question, best first: at most top_k, and only
    those at or above min_similarity (an unrelated table is never returned just because it is the nearest)."""
    query_vec = embed_query(query_text)
    with _borrow("search_tables", top_k=top_k, min_similarity=min_similarity) as (conn, span):
        rows = conn.execute(
            f"""SELECT table_id, text, caption, page, parent_id, source_pdf, 1 - (embedding <=> %s) AS similarity
                FROM {SCHEMA_NAME}.doc_tables
                ORDER BY embedding <=> %s
                LIMIT %s""",
            (query_vec, query_vec, top_k),
        ).fetchall()
        kept = [r for r in rows if float(r[6]) >= min_similarity]
        span.update(metadata={"rows": len(rows), "above_threshold": len(kept)})
    return [_table_dict(r[:6], r[6]) for r in kept]
