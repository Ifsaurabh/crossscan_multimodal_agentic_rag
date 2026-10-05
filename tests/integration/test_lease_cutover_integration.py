"""The worker lease and the cutover SQL against a REAL Postgres (throwaway schemas, dropped at the end).

Run:  python -m pytest tests/integration/test_lease_cutover_integration.py --run-integration
Nothing outside the throwaway `it_...` schemas is touched (see conftest.py)."""
import os
import uuid

import psycopg
import pytest

from ingestion import pipeline_schema, worker_lease as wl
from retrieval import chat_store, cutover as co, query_cache, setup_app_db

pytestmark = pytest.mark.integration


# ---------- the lease ----------

@pytest.fixture
def lease_table(schema, conn, monkeypatch):
    assert schema.startswith("it_")
    monkeypatch.setattr(wl, "SCHEMA_NAME", schema)
    conn.execute(wl.setup_sql(schema))
    conn.execute(f"DELETE FROM {schema}.worker_lease")
    conn.commit()
    return conn


def test_only_one_holder_gets_the_lease_at_a_time(lease_table):
    c = lease_table
    assert wl.try_acquire(c, "run-a") is True
    assert wl.try_acquire(c, "run-b") is False           # taken: the second execution exits at once
    assert wl.try_acquire(c, "run-a") is True            # the holder asking again still holds it


def test_only_the_holder_can_renew_and_release(lease_table, schema):
    c = lease_table
    wl.try_acquire(c, "run-a")
    assert wl.renew(c, "run-a") is True and wl.renew(c, "run-b") is False
    wl.release(c, "run-b")                                # not ours: ignored
    assert c.execute(f"SELECT holder FROM {schema}.worker_lease").fetchone()[0] == "run-a"
    wl.release(c, "run-a")
    assert c.execute(f"SELECT COUNT(*) FROM {schema}.worker_lease").fetchone()[0] == 0
    assert wl.try_acquire(c, "run-b") is True            # free again


def test_a_lease_that_ran_out_is_taken_over(lease_table, schema):
    c = lease_table
    wl.try_acquire(c, "dead-run")
    c.execute(f"UPDATE {schema}.worker_lease SET expires_at = now() - interval '1 second'")  # the holder died
    assert wl.try_acquire(c, "run-b") is True
    assert wl.renew(c, "dead-run") is False              # the old holder finds it lost
    assert c.execute(f"SELECT holder FROM {schema}.worker_lease").fetchone()[0] == "run-b"


def test_the_lease_function_holds_renews_and_gives_back(lease_table, schema):
    from shared import db

    with wl.hold(connect=db.connection, holder="me", renew_every=0.2) as lease:
        assert lease.acquired
        with wl.hold(connect=db.connection, holder="other") as second:
            assert second.acquired is False               # a second execution while the first works
    with db.connection() as c:
        assert c.execute(f"SELECT COUNT(*) FROM {schema}.worker_lease").fetchone()[0] == 0


# ---------- the cutover ----------

@pytest.fixture
def target(schema):
    from shared import db

    name = f"it_{uuid.uuid4().hex[:10]}"
    assert name.startswith("it_") and name != schema
    url = os.environ["DATABASE_URL"]
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {name}")
        admin.execute(f"SET search_path TO {name}, public")
        admin.execute(setup_app_db.SCHEMA_SQL.replace(f"{setup_app_db.SCHEMA_NAME}.", f"{name}."))
        admin.execute(query_cache.setup_sql(name))
    try:
        yield name
    finally:
        assert name.startswith("it_")
        with psycopg.connect(url, autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA {name} CASCADE")


def test_the_app_data_is_copied_with_its_ids_and_the_counters_continue_after_them(schema, target, conn, make_user):
    user = make_user()
    session_id = chat_store.create_session(conn, user["user_id"])
    first = chat_store.add_message(conn, user["user_id"], session_id, "user", "hello")
    second = chat_store.add_message(conn, user["user_id"], session_id, "assistant", "hi", metadata={"sources": []})
    conn.commit()

    plan = {t: (s, n) for t, s, n in co.plan_copy(conn, schema, target)}
    assert plan["users"][0] >= 1 and plan["chat_messages"] == (2, 0)

    copied = co.copy_app_data(conn, schema, target)
    conn.commit()

    assert copied["chat_messages"] == 2
    rows = conn.execute(f"SELECT message_id, role, content FROM {target}.chat_messages ORDER BY message_id").fetchall()
    assert [r[0] for r in rows] == [first, second] and [r[1] for r in rows] == ["user", "assistant"]   # the ids are kept
    assert conn.execute(f"SELECT COUNT(*) FROM {target}.users WHERE user_id = %s", (user["user_id"],)).fetchone()[0] == 1
    # a message written to the target afterwards must not reuse an id
    new_id = conn.execute(
        f"INSERT INTO {target}.chat_messages (session_id, role, content) VALUES (%s, 'user', 'later') RETURNING message_id",
        (session_id,)).fetchone()[0]
    conn.commit()
    assert new_id > second
    # the source is untouched
    assert conn.execute(f"SELECT COUNT(*) FROM {schema}.chat_messages").fetchone()[0] == 2


def test_a_second_copy_is_refused_unless_allowed_and_then_adds_nothing_twice(schema, target, conn, make_user):
    make_user()
    conn.commit()
    co.copy_app_data(conn, schema, target)
    conn.commit()

    with pytest.raises(co.CutoverError):
        co.copy_app_data(conn, schema, target)
    conn.rollback()
    before = conn.execute(f"SELECT COUNT(*) FROM {target}.users").fetchone()[0]
    co.copy_app_data(conn, schema, target, allow_nonempty=True)
    conn.commit()
    assert conn.execute(f"SELECT COUNT(*) FROM {target}.users").fetchone()[0] == before


def test_the_cache_flush_empties_the_target_cache_only(schema, target, conn):
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:  # several statements at once: not on the test's own connection
        admin.execute(f"SET search_path TO {schema}, public")
        admin.execute(query_cache.setup_sql(schema))
    for s in (schema, target):
        conn.execute(f"INSERT INTO {s}.query_cache (query_hash, query_text, chunks_retrieved, answer) VALUES ('h', 'q', '[]', 'a') "
                     "ON CONFLICT DO NOTHING")
    conn.commit()

    assert co.flush_cache(conn, target) == 1
    conn.commit()
    assert conn.execute(f"SELECT COUNT(*) FROM {target}.query_cache").fetchone()[0] == 0
    assert conn.execute(f"SELECT COUNT(*) FROM {schema}.query_cache").fetchone()[0] == 1
