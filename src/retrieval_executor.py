import threading

from sentence_transformers import SentenceTransformer

from db import connection, SCHEMA_NAME
from graph_db import get_driver
from embedding_config import TEXT_MODEL_NAME
from retrieval_config import VECTOR_TOP_K, GRAPH_TOP_K

_model = None
# A background warm-up and a first question can ask for the model at the same
# moment; the lock makes the second caller wait for the load instead of
# starting a duplicate one.
_model_lock = threading.Lock()


def get_embedding_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = SentenceTransformer(TEXT_MODEL_NAME)
    return _model


def embed_query(query_text: str):
    model = get_embedding_model()
    return model.encode(query_text, normalize_embeddings=True)


def vector_search(query_text: str, domain: str = None, source_pdfs: list = None, top_k: int = VECTOR_TOP_K):
    """Simple semantic-only vector search over text_chunks, joined with parent
    text for generation context. Optionally narrowed by domain or a list of
    source_pdf values (used for graph-narrowed 'both' retrieval)."""
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

    with connection() as conn:
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

    return [
        {
            "chunk_id": r[0], "text": r[1], "source_pdf": r[2], "section": r[3],
            "page_start": r[4], "page_end": r[5], "parent_id": r[6],
            "parent_text": r[7], "similarity": float(r[8]),
        }
        for r in rows
    ]


def hybrid_vector_search(query_text: str, domain: str = None, top_k: int = VECTOR_TOP_K):
    """Semantic (pgvector) + keyword (Postgres full-text) combined via
    reciprocal rank fusion. Vector-only - graph 'hybrid' was explicitly
    rejected in design (no semantic signal on graph nodes to combine)."""
    query_vec = embed_query(query_text)
    fetch_k = top_k * 3

    where_domain = "AND c.domain = %s" if domain else ""

    semantic_params = [query_vec]
    if domain:
        semantic_params.append(domain)
    semantic_params += [query_vec, fetch_k]

    with connection() as conn:
        cur = conn.execute(
            f"""SELECT c.chunk_id, 1 - (c.embedding <=> %s) AS similarity
                FROM {SCHEMA_NAME}.text_chunks c
                WHERE TRUE {where_domain}
                ORDER BY c.embedding <=> %s
                LIMIT %s""",
            semantic_params,
        )
        semantic_ranked = [row[0] for row in cur.fetchall()]

        cur = conn.execute(
            f"""SELECT c.chunk_id
                FROM {SCHEMA_NAME}.text_chunks c
                WHERE c.text_search @@ plainto_tsquery('english', %s) {where_domain}
                ORDER BY ts_rank_cd(c.text_search, plainto_tsquery('english', %s)) DESC
                LIMIT %s""",
            [query_text] + ([domain] if domain else []) + [query_text, fetch_k],
        )
        keyword_ranked = [row[0] for row in cur.fetchall()]

        rrf_scores = {}
        k = 60
        for rank, chunk_id in enumerate(semantic_ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        for rank, chunk_id in enumerate(keyword_ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        top_chunk_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        if not top_chunk_ids:
            return []

        cur = conn.execute(
            f"""SELECT c.chunk_id, c.text, c.source_pdf, c.section, c.page_start, c.page_end,
                       c.parent_id, p.text AS parent_text
                FROM {SCHEMA_NAME}.text_chunks c
                JOIN {SCHEMA_NAME}.text_parents p ON c.parent_id = p.parent_id
                WHERE c.chunk_id = ANY(%s)""",
            (top_chunk_ids,),
        )
        rows = {r[0]: r for r in cur.fetchall()}

    return [
        {
            "chunk_id": cid, "text": rows[cid][1], "source_pdf": rows[cid][2],
            "section": rows[cid][3], "page_start": rows[cid][4], "page_end": rows[cid][5],
            "parent_id": rows[cid][6], "parent_text": rows[cid][7], "rrf_score": rrf_scores[cid],
        }
        for cid in top_chunk_ids if cid in rows
    ]


def graph_search(query_text: str, top_k: int = GRAPH_TOP_K):
    """Matches Method/Dataset/Metric entity names against the query text
    (substring match, case-insensitive) and traverses to connected Papers."""
    query_lower = query_text.lower()

    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """MATCH (e)<-[r]-(p:Paper)
               WHERE (e:Method OR e:Dataset OR e:Metric)
               RETURN e.name AS entity, labels(e)[0] AS entity_type, type(r) AS relationship,
                      p.source_pdf AS source_pdf, p.domain AS domain
               LIMIT 1000"""
        )
        all_rows = [dict(r) for r in result]

    matched = [
        r for r in all_rows
        if r["entity"].lower() in query_lower or any(
            word in r["entity"].lower() for word in query_lower.split() if len(word) > 2
        )
    ]

    return matched[:top_k]


def narrow_papers_from_graph(graph_results: list) -> list:
    return list({r["source_pdf"] for r in graph_results})


def get_images_for_sections(chunk_ids: list = None, source_pdfs: list = None):
    if not chunk_ids and not source_pdfs:
        return []

    driver = get_driver()
    with driver.session() as session:
        if chunk_ids:
            result = session.run(
                """MATCH (i:Image)-[:NEAR_SECTION]->(s:Section)
                   WHERE s.chunk_id IN $chunk_ids
                   RETURN DISTINCT i.image_file AS image_file, i.page AS page, s.chunk_id AS chunk_id""",
                chunk_ids=chunk_ids,
            )
        else:
            result = session.run(
                """MATCH (i:Image)-[:APPEARS_IN]->(p:Paper)
                   WHERE p.source_pdf IN $source_pdfs
                   RETURN DISTINCT i.image_file AS image_file, i.page AS page, p.source_pdf AS source_pdf""",
                source_pdfs=source_pdfs,
            )
        images = [dict(r) for r in result]
    return images


def get_tables_for_sections(chunk_ids: list = None, source_pdfs: list = None):
    """Same shape as get_images_for_sections() - Table nodes store their own
    text (unlike Image, which just points at a file), so it's returned here
    directly, not just a filename."""
    if not chunk_ids and not source_pdfs:
        return []

    driver = get_driver()
    with driver.session() as session:
        if chunk_ids:
            result = session.run(
                """MATCH (t:Table)-[:NEAR_SECTION]->(s:Section)
                   MATCH (t)-[:APPEARS_IN]->(p:Paper)
                   WHERE s.chunk_id IN $chunk_ids
                   RETURN DISTINCT t.table_id AS table_id, t.text AS text, t.page AS page,
                          s.chunk_id AS chunk_id, p.source_pdf AS source_pdf""",
                chunk_ids=chunk_ids,
            )
        else:
            result = session.run(
                """MATCH (t:Table)-[:APPEARS_IN]->(p:Paper)
                   WHERE p.source_pdf IN $source_pdfs
                   RETURN DISTINCT t.table_id AS table_id, t.text AS text, t.page AS page,
                          p.source_pdf AS source_pdf""",
                source_pdfs=source_pdfs,
            )
        tables = [dict(r) for r in result]
    return tables
