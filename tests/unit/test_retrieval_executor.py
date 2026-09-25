"""retrieval_executor: the search functions (Postgres pool + shared Neo4j driver) and the
lazily loaded embedding model. No real model, database or network is used."""
import threading
import time

import pytest

import retrieval_executor as rex
from fake_db import FakeConn

VECTOR = [0.5]


class FakeBorrow:
    """Stands in for db.connection(): lends one FakeConn, records how each borrow ended."""

    def __init__(self, conn=None):
        self.conn = conn or FakeConn()
        self.entered = 0
        self.exit_exceptions = []

    def __call__(self):
        return self

    def __enter__(self):
        self.entered += 1
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        self.exit_exceptions.append(exc_type)
        return False


class FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return list(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeDriver:
    def __init__(self, rows=()):
        self.session_obj = FakeSession(rows)
        self.closed = 0

    def session(self):
        return self.session_obj

    def close(self):
        self.closed += 1


def use_borrow(monkeypatch, conn=None):
    pool = FakeBorrow(conn)
    monkeypatch.setattr(rex, "connection", pool)
    monkeypatch.setattr(rex, "embed_query", lambda text: VECTOR)
    return pool


def use_driver(monkeypatch, rows=()):
    driver = FakeDriver(rows)
    monkeypatch.setattr(rex, "get_driver", lambda: driver)
    return driver


def chunk_row(chunk_id, similarity=None):
    base = (chunk_id, f"text of {chunk_id}", "a.pdf", "Results", 3, 4, f"{chunk_id}_parent", f"parent of {chunk_id}")
    return base + (similarity,) if similarity is not None else base


# ---------- the embedding model ----------

def test_the_embedding_model_is_loaded_once_even_when_many_threads_ask_at_once(monkeypatch):
    loads = []

    class SlowModel:
        def __init__(self, name):
            time.sleep(0.05)  # long enough for the other threads to arrive while it loads
            loads.append(name)

    monkeypatch.setattr(rex, "SentenceTransformer", SlowModel)
    monkeypatch.setattr(rex, "_model", None)
    results = []
    threads = [threading.Thread(target=lambda: results.append(rex.get_embedding_model())) for _ in range(8)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert loads == [rex.TEXT_MODEL_NAME]                     # loaded once, not eight times
    assert len(results) == 8 and all(r is results[0] for r in results)


def test_an_already_loaded_model_is_reused(monkeypatch):
    model = object()
    monkeypatch.setattr(rex, "_model", model)
    monkeypatch.setattr(rex, "SentenceTransformer", lambda name: pytest.fail("must not reload"))

    assert rex.get_embedding_model() is model


def test_a_query_is_embedded_normalised(monkeypatch):
    calls = []

    class Model:
        def encode(self, text, normalize_embeddings=False):
            calls.append((text, normalize_embeddings))
            return [0.1, 0.2]

    monkeypatch.setattr(rex, "_model", Model())

    assert rex.embed_query("what is a CNN") == [0.1, 0.2]
    assert calls == [("what is a CNN", True)]


# ---------- vector_search (Postgres) ----------

def test_vector_search_returns_chunks_with_their_parent_text_and_similarity(monkeypatch):
    conn = FakeConn(responses=[("text_chunks", [chunk_row("c1", 0.91), chunk_row("c2", 0.55)])])
    use_borrow(monkeypatch, conn)

    results = rex.vector_search("cnn accuracy")

    assert [r["chunk_id"] for r in results] == ["c1", "c2"]
    assert results[0] == {
        "chunk_id": "c1", "text": "text of c1", "source_pdf": "a.pdf", "section": "Results",
        "page_start": 3, "page_end": 4, "parent_id": "c1_parent", "parent_text": "parent of c1", "similarity": 0.91,
    }
    assert isinstance(results[0]["similarity"], float)


def test_vector_search_with_no_hits_returns_an_empty_list(monkeypatch):
    use_borrow(monkeypatch)

    assert rex.vector_search("nothing matches") == []


def test_vector_search_without_filters_has_no_where_clause(monkeypatch):
    pool = use_borrow(monkeypatch)

    rex.vector_search("q", top_k=7)

    sql, params = pool.conn.executed[0]
    assert "WHERE" not in sql
    assert params == [VECTOR, VECTOR, 7]


def test_vector_search_can_narrow_by_domain_and_by_papers(monkeypatch):
    pool = use_borrow(monkeypatch)

    rex.vector_search("q", domain="lung", source_pdfs=["a.pdf", "b.pdf"], top_k=5)

    sql, params = pool.conn.executed[0]
    assert "c.domain = %s" in sql and "c.source_pdf = ANY(%s)" in sql
    assert params == [VECTOR, "lung", ["a.pdf", "b.pdf"], VECTOR, 5]


def test_vector_search_borrows_one_connection_and_hands_it_back(monkeypatch):
    pool = use_borrow(monkeypatch)

    rex.vector_search("q")

    assert pool.entered == 1 and pool.exit_exceptions == [None]


def test_vector_search_passes_a_database_failure_to_the_pool_and_raises(monkeypatch):
    class BrokenConn(FakeConn):
        def execute(self, sql, params=None):
            raise RuntimeError("connection lost")

    pool = use_borrow(monkeypatch, BrokenConn())

    with pytest.raises(RuntimeError):
        rex.vector_search("q")

    assert pool.exit_exceptions == [RuntimeError]  # so the pool rolls the connection back


# ---------- hybrid_vector_search (semantic + keyword, fused by RRF) ----------

HYBRID_RESPONSES = [
    ("AS similarity", [("a", 0.9), ("b", 0.8), ("c", 0.7)]),        # semantic ranking a, b, c
    ("text_search @@", [("c",), ("a",), ("d",)]),                   # keyword ranking c, a, d
    ("WHERE c.chunk_id = ANY", [chunk_row("c"), chunk_row("a"), chunk_row("b")]),
]


def test_hybrid_search_fuses_both_rankings_by_reciprocal_rank(monkeypatch):
    use_borrow(monkeypatch, FakeConn(responses=HYBRID_RESPONSES))

    results = rex.hybrid_vector_search("cnn", top_k=3)

    # a: rank 0 semantic + rank 1 keyword; c: rank 2 semantic + rank 0 keyword; b: only semantic
    assert [r["chunk_id"] for r in results] == ["a", "c", "b"]
    assert results[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert results[1]["rrf_score"] == pytest.approx(1 / 63 + 1 / 61)
    assert results[2]["rrf_score"] == pytest.approx(1 / 62)


def test_hybrid_search_returns_no_more_than_top_k(monkeypatch):
    use_borrow(monkeypatch, FakeConn(responses=HYBRID_RESPONSES))

    assert len(rex.hybrid_vector_search("cnn", top_k=2)) == 2


def test_hybrid_search_skips_a_chunk_that_disappeared_before_it_was_fetched(monkeypatch):
    responses = HYBRID_RESPONSES[:2] + [("WHERE c.chunk_id = ANY", [chunk_row("a"), chunk_row("b")])]  # c is gone
    use_borrow(monkeypatch, FakeConn(responses=responses))

    assert [r["chunk_id"] for r in rex.hybrid_vector_search("cnn", top_k=3)] == ["a", "b"]


def test_hybrid_search_with_no_hits_stops_before_the_fetch_query(monkeypatch):
    pool = use_borrow(monkeypatch)

    assert rex.hybrid_vector_search("nothing") == []
    assert len(pool.conn.executed) == 2  # semantic + keyword only
    assert pool.entered == 1 and pool.exit_exceptions == [None]


def test_hybrid_search_applies_the_domain_to_both_searches(monkeypatch):
    pool = use_borrow(monkeypatch)

    rex.hybrid_vector_search("cnn", domain="lung", top_k=3)

    assert pool.conn.find("AS similarity")[0][1] == [VECTOR, "lung", VECTOR, 9]     # fetches 3x top_k candidates
    assert pool.conn.find("text_search @@")[0][1] == ["cnn", "lung", "cnn", 9]


def test_hybrid_search_uses_one_connection_for_all_three_queries(monkeypatch):
    pool = use_borrow(monkeypatch, FakeConn(responses=HYBRID_RESPONSES))

    rex.hybrid_vector_search("cnn", top_k=3)

    assert pool.entered == 1 and len(pool.conn.executed) == 3


# ---------- graph_search (Neo4j entities) ----------

ENTITY_ROWS = [
    {"entity": "VGG16", "entity_type": "Method", "relationship": "USES_METHOD", "source_pdf": "a.pdf", "domain": "lung"},
    {"entity": "LIDC-IDRI", "entity_type": "Dataset", "relationship": "USES_DATASET", "source_pdf": "a.pdf", "domain": "lung"},
    {"entity": "Accuracy", "entity_type": "Metric", "relationship": "EVALUATED_WITH", "source_pdf": "b.pdf", "domain": "lung"},
]


def test_graph_search_matches_entities_named_in_the_query(monkeypatch):
    use_driver(monkeypatch, ENTITY_ROWS)

    results = rex.graph_search("what accuracy did VGG16 get")

    assert [r["entity"] for r in results] == ["VGG16", "Accuracy"]


def test_graph_search_also_matches_on_a_word_inside_an_entity_name(monkeypatch):
    use_driver(monkeypatch, ENTITY_ROWS)

    assert [r["entity"] for r in rex.graph_search("lidc data")] == ["LIDC-IDRI"]


def test_graph_search_with_no_match_returns_nothing(monkeypatch):
    use_driver(monkeypatch, ENTITY_ROWS)

    assert rex.graph_search("weather in paris") == []


def test_graph_search_respects_top_k(monkeypatch):
    use_driver(monkeypatch, ENTITY_ROWS)

    assert len(rex.graph_search("what accuracy did VGG16 get", top_k=1)) == 1


def test_graph_search_uses_the_shared_driver_and_never_closes_it(monkeypatch):
    driver = use_driver(monkeypatch, ENTITY_ROWS)

    rex.graph_search("VGG16")
    rex.graph_search("accuracy")

    assert len(driver.session_obj.calls) == 2 and driver.closed == 0


# ---------- get_images_for_sections ----------

def test_images_with_no_ids_return_nothing_without_touching_the_database(monkeypatch):
    monkeypatch.setattr(rex, "get_driver", lambda: pytest.fail("must not connect"))

    assert rex.get_images_for_sections() == []
    assert rex.get_images_for_sections(chunk_ids=[], source_pdfs=[]) == []


def test_images_near_retrieved_sections_are_looked_up_by_chunk_id(monkeypatch):
    driver = use_driver(monkeypatch, [{"image_file": "fig1.png", "page": 3, "chunk_id": "c1"}])

    images = rex.get_images_for_sections(chunk_ids=["c1", "c2"])

    query, params = driver.session_obj.calls[0]
    assert "NEAR_SECTION" in query and params == {"chunk_ids": ["c1", "c2"]}
    assert images == [{"image_file": "fig1.png", "page": 3, "chunk_id": "c1"}]


def test_images_can_be_looked_up_by_paper_instead(monkeypatch):
    driver = use_driver(monkeypatch)

    rex.get_images_for_sections(source_pdfs=["a.pdf"])

    query, params = driver.session_obj.calls[0]
    assert "APPEARS_IN" in query and params == {"source_pdfs": ["a.pdf"]}


def test_chunk_ids_win_when_both_are_given_and_the_driver_stays_open(monkeypatch):
    driver = use_driver(monkeypatch)

    rex.get_images_for_sections(chunk_ids=["c1"], source_pdfs=["a.pdf"])

    assert driver.session_obj.calls[0][1] == {"chunk_ids": ["c1"]}
    assert driver.closed == 0


# ---------- get_tables_for_sections (new) ----------

TABLE_ROW = {"table_id": "a.pdf_table1", "text": "Model  Precision  Recall\nVGG16  0.98  0.97", "page": 14,
             "chunk_id": "c1", "source_pdf": "a.pdf"}


def test_tables_with_no_ids_return_nothing_without_touching_the_database(monkeypatch):
    monkeypatch.setattr(rex, "get_driver", lambda: pytest.fail("must not connect"))

    assert rex.get_tables_for_sections() == []
    assert rex.get_tables_for_sections(chunk_ids=[], source_pdfs=None) == []


def test_tables_near_retrieved_sections_come_back_with_their_own_text(monkeypatch):
    driver = use_driver(monkeypatch, [TABLE_ROW])

    tables = rex.get_tables_for_sections(chunk_ids=["c1"])

    query, params = driver.session_obj.calls[0]
    assert "(t:Table)-[:NEAR_SECTION]->(s:Section)" in query and params == {"chunk_ids": ["c1"]}
    assert tables == [TABLE_ROW]
    assert "VGG16" in tables[0]["text"]  # the table text lives on the node, no second lookup


def test_the_table_lookup_always_returns_the_source_paper(monkeypatch):
    """Regression: the chunk_ids branch once returned no source_pdf, so the answer prompt
    would have labelled every table 'unknown'."""
    driver = use_driver(monkeypatch)

    rex.get_tables_for_sections(chunk_ids=["c1"])
    rex.get_tables_for_sections(source_pdfs=["a.pdf"])

    for query, _ in driver.session_obj.calls:
        assert "source_pdf AS source_pdf" in query


def test_tables_can_be_looked_up_by_paper_instead(monkeypatch):
    driver = use_driver(monkeypatch)

    rex.get_tables_for_sections(source_pdfs=["a.pdf", "b.pdf"])

    query, params = driver.session_obj.calls[0]
    assert "(t:Table)-[:APPEARS_IN]->(p:Paper)" in query and params == {"source_pdfs": ["a.pdf", "b.pdf"]}


def test_chunk_ids_win_over_papers_for_tables_and_the_driver_stays_open(monkeypatch):
    driver = use_driver(monkeypatch)

    rex.get_tables_for_sections(chunk_ids=["c1"], source_pdfs=["a.pdf"])

    assert driver.session_obj.calls[0][1] == {"chunk_ids": ["c1"]}
    assert driver.closed == 0


def test_a_paper_with_no_tables_gives_an_empty_list(monkeypatch):
    use_driver(monkeypatch, [])

    assert rex.get_tables_for_sections(chunk_ids=["c1"]) == []
