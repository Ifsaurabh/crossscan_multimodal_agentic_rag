import query_cache as qc


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
        self.committed = False

    def execute(self, sql, params=None):
        if "INSERT INTO" in sql:
            query_hash, query_text, chunks_json, answer = params
            self.store[query_hash] = (chunks_json, answer)
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
