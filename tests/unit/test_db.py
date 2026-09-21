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


def test_search_path_is_set_before_register_vector_is_called(monkeypatch):
    """Regression test: register_vector() looks up the "vector" type unqualified, which only
    resolves if "public" is on the search_path. A managed-Postgres role with an EMPTY default
    search_path (seen live on Neon) raised psycopg.ProgrammingError until this order was fixed -
    a local Postgres install never caught it because its default search_path already includes
    "public"."""
    timeline = []

    class TrackingConnection(FakeConnection):
        def execute(self, sql, params=None):
            if "SET search_path" in sql:
                timeline.append("set_search_path")
            return super().execute(sql, params)

    fake_conn = TrackingConnection()
    monkeypatch.setattr(db.psycopg, "connect", lambda url: fake_conn)
    monkeypatch.setattr(db, "register_vector", lambda conn: timeline.append("register_vector"))
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")

    db.get_connection()

    assert timeline == ["set_search_path", "register_vector"]
