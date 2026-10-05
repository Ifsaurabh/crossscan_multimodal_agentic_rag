from retrieval import query_cache as qc


def test_hash_query_is_deterministic_and_case_insensitive():
    h1 = qc.hash_query("What accuracy did the CNN model achieve?")
    h2 = qc.hash_query("what accuracy did the cnn model achieve?  ")
    assert h1 == h2


def test_hash_query_differs_for_different_queries():
    h1 = qc.hash_query("What accuracy did the CNN model achieve?")
    h2 = qc.hash_query("Which papers use YOLOv11?")
    assert h1 != h2


class FakeConnection:
    def __init__(self):
        self.store = {}
        self.sources = {}
        self.committed = False

    def execute(self, sql, params=None):
        if "INSERT INTO" in sql:
            query_hash, query_text, chunks_json, answer, sources = params
            self.store[query_hash] = (chunks_json, answer)
            self.sources[query_hash] = sources
            return self
        if "SELECT" in sql:
            query_hash = params[0]
            self._last_result = self.store.get(query_hash)
            return self
        return self

    def fetchone(self):
        if self._last_result is None:
            return None
        return self._last_result

    def commit(self):
        self.committed = True


def test_write_and_get_cached_round_trip():
    conn = FakeConnection()
    qc.write_cache(conn, "What accuracy?", [{"chunk_id": "c1"}], "98% accuracy")

    result = qc.get_cached(conn, "What accuracy?")

    assert result is not None
    assert result["answer"] == "98% accuracy"
    assert conn.committed


def test_get_cached_returns_none_for_unseen_query():
    conn = FakeConnection()
    result = qc.get_cached(conn, "Never asked before")
    assert result is None


def test_the_documents_an_answer_was_built_from_are_stored_with_it():
    conn = FakeConnection()
    chunks = [{"chunk_id": "c1", "source_pdf": "lung-cancer/b.pdf"}, {"chunk_id": "c2", "source_pdf": "land-cover/a.pdf"},
              {"chunk_id": "c3", "source_pdf": "lung-cancer/b.pdf"}, {"chunk_id": "c4"}]

    qc.write_cache(conn, "What accuracy?", chunks, "98%")

    assert conn.sources[qc.hash_query("What accuracy?")] == ["land-cover/a.pdf", "lung-cancer/b.pdf"]  # distinct, sorted


def test_sources_of_handles_no_chunks_and_odd_entries():
    assert qc.sources_of([]) == [] and qc.sources_of(None) == [] and qc.sources_of(["x", {"source_pdf": ""}]) == []


def test_the_setup_adds_the_column_and_its_index_without_dropping_anything():
    sql = qc.setup_sql("it_x")
    assert "ADD COLUMN IF NOT EXISTS sources TEXT[]" in sql and "USING gin (sources)" in sql
    assert "DROP" not in sql.upper()
