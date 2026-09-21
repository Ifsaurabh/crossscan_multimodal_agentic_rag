"""Fixtures for the opt-in integration tests (real Postgres, throwaway schema).

Run:  python -m pytest tests/integration --run-integration

Safety: nothing here touches your real `rag_new` data. A schema named
`it_<random>` is created for the session, the app tables are created inside it,
every app module is pointed at it, and it is DROPPED at the end (only a schema
whose name starts with `it_` can ever be dropped).
"""
import importlib
import os
import uuid

import psycopg
import pytest

# Modules that build SQL from a module-level SCHEMA_NAME at call time.
APP_MODULES = ("db", "auth", "quotas", "chat_store", "memory", "feedback", "online_eval", "online_report")


@pytest.fixture(scope="session")
def schema():
    import db  # loads .env, so DATABASE_URL is available
    import setup_app_db

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is not set")

    name = f"it_{uuid.uuid4().hex[:10]}"
    real_schema = db.SCHEMA_NAME

    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
        admin.execute(f"CREATE SCHEMA {name}")
        admin.execute(f"SET search_path TO {name}, public")
        admin.execute(setup_app_db.SCHEMA_SQL.replace(f"{real_schema}.", f"{name}."))

    patch = pytest.MonkeyPatch()
    for module_name in APP_MODULES:
        patch.setattr(importlib.import_module(module_name), "SCHEMA_NAME", name)
    try:
        yield name
    finally:
        patch.undo()
        assert name.startswith("it_"), "refusing to drop a schema this fixture did not create"
        with psycopg.connect(url, autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA {name} CASCADE")


@pytest.fixture
def conn(schema):
    """A real connection whose search_path is the throwaway schema, with every
    table emptied first so each test starts clean."""
    import db

    connection = db.get_connection()
    connection.execute(f"TRUNCATE {schema}.users CASCADE")
    connection.commit()
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def make_user(conn, faker):
    """Creates a real user with Faker data; returns the user dict plus the
    plaintext password that was used."""
    import auth
    import factories

    def _make(role: str = "user", password: str = None) -> dict:
        plain = password or factories.password(faker)
        created = auth.create_user(conn, factories.username(faker), plain, role=role)
        return {**auth.get_user(conn, created["username"]), "password": plain}

    return _make
