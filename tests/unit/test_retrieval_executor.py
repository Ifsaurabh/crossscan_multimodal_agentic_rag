"""retrieval_executor: the search functions (Postgres pool) and the
lazily loaded embedding model. No real model, database or network is used."""
import threading
import time

import pytest

from retrieval import retrieval_executor as rex
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


def use_borrow(monkeypatch, conn=None):
    pool = FakeBorrow(conn)
    monkeypatch.setattr(rex, "connection", pool)
    monkeypatch.setattr(rex, "embed_query", lambda text: VECTOR)
    return pool


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


# ---------- the entity graph search is gone ----------

# ---------- get_images_for_parents ----------

def test_images_with_no_parent_ids_return_nothing_without_touching_the_database(monkeypatch):
    pool = use_borrow(monkeypatch)

    assert rex.get_images_for_parents() == []
    assert rex.get_images_for_parents(parent_ids=[]) == []
    assert pool.entered == 0


def test_images_are_found_by_the_parent_id_they_were_linked_to_at_ingestion(monkeypatch):
    conn = FakeConn(responses=[("images", [("lung-cancer/a/page3_img1.png", 3, "p1", "lung-cancer/a.pdf")])])
    pool = use_borrow(monkeypatch, conn)

    images = rex.get_images_for_parents(parent_ids=["p1", "p2"])

    sql, params = conn.executed[0]
    assert ".images" in sql
    assert "WHERE parent_id = ANY(%s)" in sql and params == (["p1", "p2"],)
    assert "page BETWEEN" not in sql  # no page-range join at search time
    assert images == [{"image_file": "lung-cancer/a/page3_img1.png", "page": 3, "parent_id": "p1",
                       "source_pdf": "lung-cancer/a.pdf"}]
    assert pool.entered == 1 and pool.exit_exceptions == [None]  # one connection, handed back


def test_no_images_for_the_parents_gives_an_empty_list(monkeypatch):
    use_borrow(monkeypatch, FakeConn())
    assert rex.get_images_for_parents(parent_ids=["p1"]) == []


# ---------- get_tables_for_parents ----------

TABLE_ROW = ("lung-cancer/a_t1", "Model  Precision  Recall" + chr(92) + "nVGG16  0.98  0.97", "Table 4: results", 14, "p1",
             "lung-cancer/a.pdf")


def test_tables_with_no_parent_ids_return_nothing_without_touching_the_database(monkeypatch):
    pool = use_borrow(monkeypatch)

    assert rex.get_tables_for_parents() == []
    assert rex.get_tables_for_parents(parent_ids=[]) == []
    assert pool.entered == 0


def test_tables_of_the_parents_come_back_with_their_text_caption_and_source(monkeypatch):
    conn = FakeConn(responses=[("doc_tables", [TABLE_ROW])])
    use_borrow(monkeypatch, conn)

    tables = rex.get_tables_for_parents(parent_ids=["p1"])

    sql, params = conn.executed[0]
    assert "WHERE parent_id = ANY(%s)" in sql and params == (["p1"],)
    assert tables == [{"table_id": "lung-cancer/a_t1", "text": TABLE_ROW[1], "caption": "Table 4: results", "page": 14,
                       "parent_id": "p1", "source_pdf": "lung-cancer/a.pdf"}]


def test_a_table_without_a_caption_has_an_empty_one(monkeypatch):
    row = TABLE_ROW[:2] + (None,) + TABLE_ROW[3:]
    use_borrow(monkeypatch, FakeConn(responses=[("doc_tables", [row])]))
    assert rex.get_tables_for_parents(parent_ids=["p1"])[0]["caption"] == ""


def test_a_parent_with_no_tables_gives_an_empty_list(monkeypatch):
    use_borrow(monkeypatch, FakeConn())
    assert rex.get_tables_for_parents(parent_ids=["p1"]) == []


# ---------- search_tables ----------

def search_row(similarity):
    return TABLE_ROW + (similarity,)


def test_the_table_search_orders_by_distance_and_asks_for_at_most_top_k(monkeypatch):
    conn = FakeConn(responses=[("doc_tables", [search_row(0.8)])])
    use_borrow(monkeypatch, conn)

    rex.search_tables("recall in Table 4", top_k=3)

    sql, params = conn.executed[0]
    assert "ORDER BY embedding <=> %s" in sql and "LIMIT %s" in sql and params == (VECTOR, VECTOR, 3)


def test_only_tables_at_or_above_the_minimum_similarity_are_returned(monkeypatch):
    rows = [search_row(0.82), search_row(0.45), search_row(0.31)]
    use_borrow(monkeypatch, FakeConn(responses=[("doc_tables", rows)]))

    found = rex.search_tables("q", min_similarity=0.45)

    assert [t["similarity"] for t in found] == [0.82, 0.45]
    assert found[0]["table_id"] == "lung-cancer/a_t1" and found[0]["caption"] == "Table 4: results"


def test_an_unrelated_nearest_table_is_not_returned(monkeypatch):
    use_borrow(monkeypatch, FakeConn(responses=[("doc_tables", [search_row(0.2)])]))
    assert rex.search_tables("training time table") == []


def test_the_default_limits_come_from_the_config(monkeypatch):
    import inspect

    from retrieval import retrieval_config as cfg

    defaults = inspect.signature(rex.search_tables).parameters
    assert defaults["top_k"].default == cfg.TABLE_TOP_K == 3
    assert defaults["min_similarity"].default == cfg.MIN_TABLE_SIMILARITY


# ---------- batch embedding and a cap on parallel searches ----------

class BatchModel:
    def __init__(self):
        self.calls = []

    def encode(self, text, normalize_embeddings=False):
        self.calls.append(text)
        return [[float(len(t))] for t in text] if isinstance(text, list) else [float(len(text))]


def test_several_texts_are_embedded_in_one_call_in_order(monkeypatch):
    model = BatchModel()
    monkeypatch.setattr(rex, "get_embedding_model", lambda: model)

    vectors = rex.embed_queries(["a", "bb", "ccc"])

    assert model.calls == [["a", "bb", "ccc"]] and vectors == [[1.0], [2.0], [3.0]]
    assert rex.embed_queries([]) == [] and len(model.calls) == 1  # nothing to embed: the model is not called


def test_a_search_given_its_vector_does_not_embed_again(monkeypatch):
    monkeypatch.setattr(rex, "embed_query", lambda text: pytest.fail("the vector was already made"))
    conn = FakeConn(responses=[("text_chunks", [chunk_row("c1", 0.9)])])
    use_borrow(monkeypatch, conn)

    rex.vector_search("q", query_vec=[0.1, 0.2])
    rex.hybrid_vector_search("q", query_vec=[0.1, 0.2])

    assert conn.executed[0][1][0] == [0.1, 0.2]


def test_no_more_searches_than_the_pool_allows_run_at_once(monkeypatch):
    import threading
    import time

    monkeypatch.setattr(rex, "_search_slots", threading.BoundedSemaphore(2))
    running, peak, lock = [0], [0], threading.Lock()

    class Conn:
        def execute(self, sql, params=None):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1
            return type("C", (), {"fetchall": lambda self: []})()

    monkeypatch.setattr(rex, "connection", FakeBorrow(Conn()))
    monkeypatch.setattr(rex, "embed_query", lambda text: VECTOR)

    threads = [threading.Thread(target=rex.vector_search, args=("q",)) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert peak[0] == 2  # six searches asked, two at a time


def test_a_failed_search_gives_its_slot_back(monkeypatch):
    import threading

    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(rex, "_search_slots", slots)

    class Down:
        def execute(self, *a, **k):
            raise RuntimeError("database unreachable")

    monkeypatch.setattr(rex, "connection", FakeBorrow(Down()))
    monkeypatch.setattr(rex, "embed_query", lambda text: VECTOR)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            rex.vector_search("q")

    assert slots.acquire(blocking=False)  # still available after three failures


def test_the_parallel_limit_is_the_pool_size():
    from retrieval import retrieval_config as cfg

    assert cfg.MAX_PARALLEL_SEARCHES == 8 or cfg.MAX_PARALLEL_SEARCHES > 0
