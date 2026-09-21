import db


class FakeConnection:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return self

    def commit(self):
        pass


def test_get_connection_enables_extension_and_sets_search_path(monkeypatch):
    fake_conn = FakeConnection()

    monkeypatch.setattr(db.psycopg, "connect", lambda url: fake_conn)
    monkeypatch.setattr(db, "register_vector", lambda conn: None)
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")

    result = db.get_connection()

    assert result is fake_conn
    sql_statements = [sql for sql, _ in fake_conn.executed]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in sql_statements)
    assert any("SET search_path" in s for s in sql_statements)
    assert any(db.SCHEMA_NAME in s for s in sql_statements if "search_path" in s)
