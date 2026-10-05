"""worker_lease (one run at a time) and cutover (the data steps of the switch). Stand-in connections; the real SQL is in the
integration tests."""
import threading
import time
from contextlib import contextmanager

import pytest

from fake_db import FakeConn
from ingestion import worker_lease as wl
from retrieval import cutover as co


# ---------- worker_lease ----------

@contextmanager
def fake_connect():
    yield object()


@pytest.fixture
def calls(monkeypatch):
    log = {"acquire": [], "renew": [], "release": [], "answers": {"acquire": True, "renew": True}}
    monkeypatch.setattr(wl, "try_acquire", lambda conn, holder, name=wl.LEASE_NAME, ttl=wl.TTL_SECONDS:
                        log["acquire"].append(holder) or log["answers"]["acquire"])
    monkeypatch.setattr(wl, "renew", lambda conn, holder, name=wl.LEASE_NAME, ttl=wl.TTL_SECONDS:
                        log["renew"].append(holder) or log["answers"]["renew"])
    monkeypatch.setattr(wl, "release", lambda conn, holder, name=wl.LEASE_NAME: log["release"].append(holder))
    return log


def test_a_run_that_gets_the_lease_holds_it_and_gives_it_back(calls):
    with wl.hold(connect=fake_connect, holder="me", renew_every=10) as lease:
        assert lease.acquired and not lease.lost.is_set()
        assert calls["release"] == []

    assert calls["acquire"] == ["me"] and calls["release"] == ["me"]


def test_a_run_that_does_not_get_the_lease_does_nothing_and_releases_nothing(calls):
    calls["answers"]["acquire"] = False

    with wl.hold(connect=fake_connect, holder="late", renew_every=0.01) as lease:
        assert lease.acquired is False
        time.sleep(0.05)

    assert calls["renew"] == [] and calls["release"] == []  # no heartbeat, and it never touches somebody else's lease


def test_the_lease_is_renewed_while_the_work_runs(calls):
    with wl.hold(connect=fake_connect, holder="me", renew_every=0.02):
        time.sleep(0.15)
    assert len(calls["renew"]) >= 2


def test_losing_the_lease_is_noticed_and_the_run_is_told_to_stop(calls):
    calls["answers"]["renew"] = False
    with wl.hold(connect=fake_connect, holder="me", renew_every=0.02) as lease:
        assert lease.lost.wait(timeout=2)


def test_the_lease_is_given_back_even_when_the_work_fails(calls):
    with pytest.raises(RuntimeError):
        with wl.hold(connect=fake_connect, holder="me", renew_every=10):
            raise RuntimeError("the pipeline crashed")
    assert calls["release"] == ["me"]


def test_a_failing_renewal_is_retried_not_fatal(monkeypatch, calls, capsys):
    attempts = []

    def flaky(conn, holder, name=wl.LEASE_NAME, ttl=wl.TTL_SECONDS):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("database hiccup")
        return True

    monkeypatch.setattr(wl, "renew", flaky)
    with wl.hold(connect=fake_connect, holder="me", renew_every=0.02) as lease:
        time.sleep(0.15)
        assert not lease.lost.is_set()
    assert len(attempts) >= 2 and "will retry" in capsys.readouterr().out


def test_every_run_has_its_own_holder_id(monkeypatch):
    monkeypatch.setenv("CLOUD_RUN_EXECUTION", "crossscan-ingest-worker-abc12")
    first, second = wl.holder_id(), wl.holder_id()
    assert first.startswith("crossscan-ingest-worker-abc12-") and first != second


def test_the_lease_table_is_only_created_by_the_pipeline_schema():
    from ingestion import pipeline_schema

    assert "it_x.worker_lease" in wl.setup_sql("it_x") and "worker_lease" in pipeline_schema.setup_sql("it_x")
    assert "DROP" not in wl.setup_sql("it_x").upper()


def test_run_worker_stops_taking_messages_when_told_to_stop():
    from ingestion import ingest_worker as iw

    class Subscriber:
        def pull(self, request, timeout):
            raise AssertionError("must not pull once told to stop")

    assert iw.run_worker(Subscriber(), "sub", lambda name: None, should_stop=lambda: True) == []


# ---------- cutover ----------

class Db(FakeConn):
    """Counts by table name: {"source.users": 5, ...}; tables not listed do not exist."""

    def __init__(self, counts=None, columns=None, null_parents=0):
        super().__init__()
        self.counts, self.columns, self.null_parents = counts or {}, columns or ["a", "b"], null_parents

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        flat = " ".join(sql.split())
        from fake_db import FakeCursor

        if "to_regclass" in flat:
            return FakeCursor([(params[0],)] if params[0] in self.counts else [(None,)])
        if flat.startswith("SELECT COUNT(*) FROM") and "parent_id IS NULL" in flat:
            return FakeCursor([(self.null_parents,)])
        if flat.startswith("SELECT COUNT(*) FROM"):
            return FakeCursor([(self.counts[flat.split("FROM ")[1].split()[0]],)])
        if "information_schema.columns" in flat and "ordinal_position" in flat:
            return FakeCursor([(c,) for c in self.columns])
        if flat.startswith("INSERT INTO"):
            cur = FakeCursor([])
            cur.rowcount = self.counts.get("source." + flat.split("FROM source.")[1].split()[0], 0) if "FROM source." in flat else 0
            return cur
        return FakeCursor([])


def counts(source=None, target=None, data=None):
    out = {}
    for table in co.APP_TABLES:
        if source is not None:
            out[f"source.{table}"] = source
        if target is not None:
            out[f"target.{table}"] = target
    out.update(data or {})
    return out


@pytest.mark.parametrize("name", ["rag_v2", "rag_new", "it_abc123"])
def test_plain_schema_names_are_accepted(name):
    assert co.check_schema_name(name) == name


@pytest.mark.parametrize("name", ["", None, "rag v2", "rag;drop schema x", "Rag", "1abc", "a.b"])
def test_anything_that_is_not_a_plain_schema_name_is_refused(name):
    with pytest.raises(co.CutoverError):
        co.check_schema_name(name)


def test_the_same_schema_cannot_be_both_source_and_target():
    with pytest.raises(co.CutoverError):
        co.plan_copy(Db(counts()), "rag_v2", "rag_v2")


def test_the_plan_lists_every_app_table_with_its_counts():
    plan = co.plan_copy(Db(counts(source=3, target=0)), "source", "target")
    assert [t for t, _, _ in plan] == list(co.APP_TABLES) and all((s, t) == (3, 0) for _, s, t in plan)


def test_a_target_that_already_holds_app_data_is_refused_and_nothing_is_copied():
    db = Db(counts(source=3, target=1))
    with pytest.raises(co.CutoverError, match="already holds rows"):
        co.copy_app_data(db, "source", "target")
    assert not [s for s, _ in db.executed if s.lstrip().startswith("INSERT")]


def test_the_copy_keeps_ids_skips_existing_rows_and_never_touches_the_source():
    db = Db(counts(source=3, target=0))

    copied = co.copy_app_data(db, "source", "target")

    inserts = [" ".join(s.split()) for s, _ in db.executed if s.lstrip().startswith("INSERT")]
    assert len(inserts) == len(co.APP_TABLES) and all("ON CONFLICT DO NOTHING" in s for s in inserts)
    assert inserts[0].startswith("INSERT INTO target.users (a, b) SELECT a, b FROM source.users")
    assert copied["users"] == 3
    assert not [s for s, _ in db.executed if s.lstrip().upper().startswith(("DELETE", "UPDATE", "DROP", "TRUNCATE"))]


def test_allow_nonempty_adds_only_the_missing_rows():
    db = Db(counts(source=3, target=1))
    assert co.copy_app_data(db, "source", "target", allow_nonempty=True)["users"] == 3


def test_the_cache_flush_empties_only_the_target_cache():
    db = Db({"target.query_cache": 1})
    co.flush_cache(db, "target")
    assert [s for s, _ in db.executed if "DELETE" in s] == ["DELETE FROM target.query_cache"]
    assert co.flush_cache(Db({}), "target") == 0  # no cache table: nothing to flush


def test_verify_says_what_is_missing_before_the_switch():
    data = {f"target.{t}": 0 for t in co.DATA_TABLES}
    report = co.verify(Db(counts(source=2, target=0, data=data)), "source", "target")
    assert any("no text chunks" in p for p in report["problems"])
    assert any("users: 2 row(s) in the source, none in the target" in p for p in report["problems"])


def test_verify_is_clean_when_the_target_is_ready():
    data = {f"target.{t}": 5 for t in co.DATA_TABLES}
    assert co.verify(Db(counts(source=2, target=2, data=data)), "source", "target")["problems"] == []


def test_verify_flags_images_and_tables_without_a_parent():
    data = {f"target.{t}": 5 for t in co.DATA_TABLES}
    report = co.verify(Db(counts(source=2, target=2, data=data), null_parents=4), "source", "target")
    assert any("image(s) have no parent" in p for p in report["problems"]) and any("table(s) have no parent" in p for p in report["problems"])


@contextmanager
def connector(db):
    yield db


def test_a_dry_run_copies_nothing(capsys):
    db = Db(counts(source=3, target=0))
    assert co.main(["--source", "source", "--target", "target", "--copy", "--dry-run"], connect=lambda: connector(db)) == 0
    assert "Dry run" in capsys.readouterr().out and not [s for s, _ in db.executed if s.lstrip().startswith("INSERT")]


def test_the_command_reports_a_refused_copy_without_a_crash(capsys):
    db = Db(counts(source=3, target=1))
    assert co.main(["--source", "source", "--target", "target", "--copy"], connect=lambda: connector(db)) == 1
    assert "Not copied" in capsys.readouterr().out


def test_without_an_action_the_command_only_shows_its_help(capsys):
    assert co.main([], connect=lambda: pytest.fail("no connection needed")) == 0
    assert "--copy" in capsys.readouterr().out
