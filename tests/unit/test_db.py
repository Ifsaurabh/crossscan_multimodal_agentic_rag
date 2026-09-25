import pytest

import db


class FakeConnection:
    def __init__(self):
        self.executed = []
        self.commits = 0

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return self

    def commit(self):
        self.commits += 1


class FakePool:
    """Stands in for psycopg_pool.ConnectionPool and records how it was built."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.opened = 0
        self.closed = 0
        FakePool.instances.append(self)

    def open(self):
        self.opened += 1

    def close(self):
        self.closed += 1

    def connection(self):
        return "a-borrowed-connection-context"

    @staticmethod
    def check_connection(conn):
        pass


@pytest.fixture(autouse=True)
def fresh_pool_state(monkeypatch):
    FakePool.instances = []
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "ConnectionPool", FakePool)
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")
    for name in ("DB_POOL_MIN", "DB_POOL_MAX", "DB_POOL_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


# ---------- direct connection (one-shot scripts) ----------

def test_get_connection_enables_extension_and_sets_search_path(monkeypatch):
    fake_conn = FakeConnection()

    monkeypatch.setattr(db.psycopg, "connect", lambda url, **kwargs: fake_conn)
    monkeypatch.setattr(db, "register_vector", lambda conn: None)

    result = db.get_connection()

    assert result is fake_conn
    sql_statements = [sql for sql, _ in fake_conn.executed]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in sql_statements)
    assert any("SET search_path" in s for s in sql_statements)
    assert any(db.SCHEMA_NAME in s for s in sql_statements if "search_path" in s)


def test_get_connection_has_a_connect_timeout(monkeypatch):
    seen = {}
    monkeypatch.setattr(db.psycopg, "connect", lambda url, **kwargs: seen.update(kwargs) or FakeConnection())
    monkeypatch.setattr(db, "register_vector", lambda conn: None)

    db.get_connection()

    assert seen == {"connect_timeout": db.CONNECT_TIMEOUT_SECONDS}


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
    monkeypatch.setattr(db.psycopg, "connect", lambda url, **kwargs: fake_conn)
    monkeypatch.setattr(db, "register_vector", lambda conn: timeline.append("register_vector"))

    db.get_connection()

    assert timeline == ["set_search_path", "register_vector"]


# ---------- pooled connections (the running application) ----------

def test_a_new_pooled_connection_is_set_up_in_the_right_order_and_left_idle(monkeypatch):
    timeline = []

    class TrackingConnection(FakeConnection):
        def execute(self, sql, params=None):
            timeline.append("set_search_path" if "SET search_path" in sql else sql)
            return super().execute(sql, params)

        def commit(self):
            timeline.append("commit")
            super().commit()

    monkeypatch.setattr(db, "register_vector", lambda conn: timeline.append("register_vector"))

    db._set_up_connection(TrackingConnection())

    # the pool needs the connection back outside a transaction, hence the commit
    assert timeline == ["set_search_path", "register_vector", "commit"]


def test_the_pool_does_not_create_the_extension_on_every_connection(monkeypatch):
    conn = FakeConnection()
    monkeypatch.setattr(db, "register_vector", lambda c: None)

    db._set_up_connection(conn)

    assert not any("CREATE EXTENSION" in sql for sql, _ in conn.executed)


def test_the_pool_is_built_once_opened_and_shared():
    first = db.get_pool()
    second = db.get_pool()

    assert first is second
    assert len(FakePool.instances) == 1
    assert first.opened == 1


def test_the_pool_uses_the_documented_defaults():
    kwargs = db.get_pool().kwargs

    assert (kwargs["min_size"], kwargs["max_size"], kwargs["timeout"]) == (2, 8, 15.0)
    assert kwargs["conninfo"] == "postgresql://fake/db"
    assert kwargs["kwargs"] == {"connect_timeout": db.CONNECT_TIMEOUT_SECONDS}


def test_the_pool_size_and_timeout_come_from_env(monkeypatch):
    monkeypatch.setenv("DB_POOL_MIN", "3")
    monkeypatch.setenv("DB_POOL_MAX", "5")
    monkeypatch.setenv("DB_POOL_TIMEOUT", "7")

    kwargs = db.get_pool().kwargs

    assert (kwargs["min_size"], kwargs["max_size"], kwargs["timeout"]) == (3, 5, 7.0)


def test_the_pool_sets_up_each_connection_and_health_checks_before_lending():
    kwargs = db.get_pool().kwargs

    assert kwargs["configure"] is db._set_up_connection
    assert kwargs["check"] == FakePool.check_connection  # replaces a connection Neon closed while idle
    assert kwargs["open"] is False  # opened explicitly, not in the constructor


def test_connection_borrows_from_the_shared_pool():
    assert db.connection() == "a-borrowed-connection-context"
    assert len(FakePool.instances) == 1


def test_closing_the_pool_closes_it_and_a_later_call_builds_a_new_one():
    first = db.get_pool()

    db.close_pool()

    assert first.closed == 1
    assert db.get_pool() is not first
    assert len(FakePool.instances) == 2


def test_closing_when_no_pool_exists_is_harmless():
    db.close_pool()
