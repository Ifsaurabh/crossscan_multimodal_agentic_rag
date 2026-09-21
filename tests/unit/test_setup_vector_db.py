import setup_vector_db as svd


class FakeConnection:
    def __init__(self):
        self.executed = []
        self.committed = False
        self.closed = False

    def execute(self, sql, params=None):
        self.executed.append(sql)
        return self

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def test_setup_vector_db_creates_schema_and_tables(monkeypatch):
    fake_conn = FakeConnection()
    monkeypatch.setattr(svd, "get_connection", lambda: fake_conn)

    svd.setup_vector_db()

    assert fake_conn.committed
    assert fake_conn.closed
    executed_sql = fake_conn.executed[0]
    assert "CREATE SCHEMA IF NOT EXISTS" in executed_sql
    assert "text_parents" in executed_sql
    assert "text_chunks" in executed_sql
    assert "images" in executed_sql
    assert "vector(768)" in executed_sql
    assert "vector(512)" in executed_sql
    assert "hnsw" in executed_sql
