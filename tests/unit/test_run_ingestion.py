"""run_ingestion: the orchestrator that runs the 12 ingestion stages in order, records every
run and stage in Postgres, stops on the first failure and resumes from it next time.

The stage modules (Docling, the embedding models, ...) are heavy, so they are replaced by
empty fakes BEFORE run_ingestion is imported; the bookkeeping database is a scripted fake."""
import importlib
import sys
import types

import pytest

import usage_tracker

STAGE_FUNCTIONS = [
    ("load_data", "load_dataset"), ("extract_text", "extract_text"), ("extract_images", "extract_images"),
    ("prepare_documents", "prepare_documents"), ("extract_entities", "extract_entities"),
    ("classify_domain", "classify_domain"), ("ingestion_guardrails", "run_guardrails"),
    ("chunk_documents", "chunk_documents"), ("embed_text", "embed_text"), ("embed_images", "embed_images"),
    ("load_vector_db", "load_vector_db"), ("load_graph_db", "load_graph_db"),
]
STAGE_NAMES = [name for name, _ in STAGE_FUNCTIONS]


@pytest.fixture
def ri(monkeypatch):
    """run_ingestion imported against fake stage modules, then removed again."""
    for module_name, function_name in STAGE_FUNCTIONS:
        monkeypatch.setitem(sys.modules, module_name, types.SimpleNamespace(**{function_name: lambda: None}))
    sys.modules.pop("run_ingestion", None)
    module = importlib.import_module("run_ingestion")
    yield module
    sys.modules.pop("run_ingestion", None)


class Cursor:
    def __init__(self, rows, returning):
        self.rows = rows
        self.description = ("col",) if returning else None

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class ScriptedDB:
    """A fake Postgres for the run/stage bookkeeping tables. Every get_connection() call
    hands out a NEW connection, so 'a fresh connection per write' can be counted."""

    def __init__(self, unfinished_runs=(), stage_rows=(), new_run_id=7):
        self.unfinished_runs = list(unfinished_runs)
        self.stage_rows = list(stage_rows)
        self.new_run_id = new_run_id
        self.statements = []
        self.opened = 0
        self.closed = 0
        self.commits = 0
        self._stage_id = 100

    def get_connection(self):
        self.opened += 1
        return ScriptedConn(self)

    def find(self, needle):
        return [(sql, params) for sql, params in self.statements if needle in sql]


class ScriptedConn:
    def __init__(self, db):
        self.db = db

    def execute(self, sql, params=()):
        db = self.db
        db.statements.append((sql, params))
        if "INSERT INTO" in sql and "ingestion_runs" in sql:
            return Cursor([(db.new_run_id,)], returning=True)
        if "INSERT INTO" in sql and "ingestion_run_stages" in sql:
            db._stage_id += 1
            return Cursor([(db._stage_id,)], returning=True)
        if "SELECT run_id" in sql:
            return Cursor(db.unfinished_runs, returning=False)
        if "SELECT stage, status" in sql:
            return Cursor(db.stage_rows, returning=False)
        return Cursor([], returning=False)

    def commit(self):
        self.db.commits += 1

    def close(self):
        self.db.closed += 1


def install(ri, monkeypatch, db, failing=None, tokens=None):
    """Replaces the stages with recorders and the database with the scripted fake."""
    calls = []

    def make(name):
        def stage():
            calls.append(name)
            if tokens and name in tokens:
                usage_tracker.record_tokens(*tokens[name])
            if name == failing:
                raise RuntimeError(f"{name} blew up")
        return stage

    monkeypatch.setattr(ri, "STAGES", [(name, make(name)) for name in STAGE_NAMES])
    monkeypatch.setattr(ri, "get_connection", db.get_connection)
    return calls


# ---------- the pipeline definition ----------

def test_the_pipeline_has_the_twelve_stages_in_the_documented_order(ri):
    assert ri.STAGE_NAMES == STAGE_NAMES
    assert [name for name, _ in ri.STAGES] == STAGE_NAMES


def test_every_stage_is_a_callable(ri):
    assert all(callable(func) for _, func in ri.STAGES)


# ---------- a clean run ----------

def test_a_clean_run_executes_every_stage_once_in_order(ri, monkeypatch):
    db = ScriptedDB()
    calls = install(ri, monkeypatch, db)

    ri.run_ingestion()

    assert calls == STAGE_NAMES


def test_a_clean_run_records_a_new_run_with_its_trigger_and_marks_it_succeeded(ri, monkeypatch):
    db = ScriptedDB(new_run_id=7)
    install(ri, monkeypatch, db)

    ri.run_ingestion(trigger="scheduled")

    assert db.find("ingestion_runs (trigger)")[0][1] == ("scheduled",)
    final_sql, final_params = db.find("SET status = 'succeeded', finished_at = now() WHERE run_id")[0]
    assert final_params == (7,)


def test_every_stage_is_recorded_as_started_then_succeeded(ri, monkeypatch):
    db = ScriptedDB()
    install(ri, monkeypatch, db)

    ri.run_ingestion()

    started = db.find("ingestion_run_stages (run_id, stage)")
    finished = db.find("ingestion_run_stages SET status = 'succeeded'")
    assert [params for _, params in started] == [(7, name) for name in STAGE_NAMES]
    assert len(finished) == len(STAGE_NAMES)


def test_the_default_trigger_is_manual(ri, monkeypatch):
    db = ScriptedDB()
    install(ri, monkeypatch, db)

    ri.run_ingestion()

    assert db.find("ingestion_runs (trigger)")[0][1] == ("manual",)


def test_the_time_and_tokens_each_stage_used_are_recorded(ri, monkeypatch):
    db = ScriptedDB()
    install(ri, monkeypatch, db, tokens={"extract_entities": (120, 30), "classify_domain": (40, 5)})

    ri.run_ingestion()

    by_stage_id = {params[3]: params for _, params in db.find("ingestion_run_stages SET status = 'succeeded'")}
    stage_ids = {name: 101 + i for i, name in enumerate(STAGE_NAMES)}
    elapsed, prompt, output, _ = by_stage_id[stage_ids["extract_entities"]]
    assert (prompt, output) == (120, 30) and elapsed >= 0
    assert by_stage_id[stage_ids["classify_domain"]][1:3] == (40, 5)
    assert by_stage_id[stage_ids["load_data"]][1:3] == (0, 0)  # a stage that used no model reports zero


def test_every_write_uses_a_fresh_connection_that_is_closed_again(ri, monkeypatch):
    """A long stage (Docling takes minutes) must never hold a connection open: Neon closes idle ones."""
    db = ScriptedDB()
    install(ri, monkeypatch, db)

    ri.run_ingestion()

    assert db.opened == db.closed and db.opened > len(STAGE_NAMES)
    assert db.commits == len(db.statements) - len(db.find("SELECT"))  # every write was committed


# ---------- a failing stage ----------

def test_the_first_failure_stops_the_run_and_later_stages_never_start(ri, monkeypatch, capsys):
    db = ScriptedDB()
    calls = install(ri, monkeypatch, db, failing="extract_images")

    ri.run_ingestion()  # returns normally: the failure is recorded, not raised

    assert calls == ["load_data", "extract_text", "extract_images"]
    assert "Stage 'extract_images' FAILED" in capsys.readouterr().out


def test_a_failed_stage_is_recorded_with_its_error_and_the_run_is_marked_failed(ri, monkeypatch):
    db = ScriptedDB(new_run_id=7)
    install(ri, monkeypatch, db, failing="extract_images")

    ri.run_ingestion()

    _, failed_params = db.find("ingestion_run_stages SET status = 'failed'")[0]
    elapsed, prompt_tokens, output_tokens, error_message, stage_id = failed_params
    assert error_message == "extract_images blew up" and elapsed >= 0
    assert stage_id == 103  # the third stage row that was inserted
    run_failed = db.find("ingestion_runs SET status = 'failed'")[0]
    assert run_failed[1] == (7,)
    assert db.find("SET status = 'succeeded', finished_at = now() WHERE run_id") == []  # never marked succeeded


def test_a_failing_first_stage_runs_nothing_else(ri, monkeypatch):
    db = ScriptedDB()
    calls = install(ri, monkeypatch, db, failing="load_data")

    ri.run_ingestion()

    assert calls == ["load_data"]


def test_a_failing_last_stage_still_fails_the_run(ri, monkeypatch):
    db = ScriptedDB()
    install(ri, monkeypatch, db, failing="load_graph_db")

    ri.run_ingestion()

    assert db.find("ingestion_runs SET status = 'failed'") and not db.find("SET status = 'succeeded', finished_at = now() WHERE run_id")


# ---------- starting from a stage / resuming ----------

def test_start_from_skips_the_earlier_stages(ri, monkeypatch):
    db = ScriptedDB()
    calls = install(ri, monkeypatch, db)

    ri.run_ingestion(start_from="chunk_documents")

    assert calls == ["chunk_documents", "embed_text", "embed_images", "load_vector_db", "load_graph_db"]


def test_an_unknown_start_stage_is_refused(ri, monkeypatch):
    install(ri, monkeypatch, ScriptedDB())

    with pytest.raises(ValueError):
        ri.run_ingestion(start_from="not_a_stage")


def test_with_no_earlier_run_the_resume_point_is_none(ri, monkeypatch):
    monkeypatch.setattr(ri, "get_connection", ScriptedDB().get_connection)

    assert ri._find_resume_point() == (None, None)


def test_an_unfinished_run_resumes_from_the_first_stage_that_did_not_succeed(ri, monkeypatch):
    db = ScriptedDB(unfinished_runs=[(5,)], stage_rows=[
        ("load_data", "succeeded"), ("extract_text", "succeeded"), ("extract_images", "failed"),
    ])
    monkeypatch.setattr(ri, "get_connection", db.get_connection)

    assert ri._find_resume_point() == (5, "extract_images")


def test_a_stage_that_ran_but_never_finished_is_redone(ri, monkeypatch):
    db = ScriptedDB(unfinished_runs=[(5,)], stage_rows=[("load_data", "succeeded"), ("extract_text", "running")])
    monkeypatch.setattr(ri, "get_connection", db.get_connection)

    assert ri._find_resume_point() == (5, "extract_text")


def test_a_run_whose_stages_all_succeeded_is_treated_as_done(ri, monkeypatch):
    db = ScriptedDB(unfinished_runs=[(5,)], stage_rows=[(name, "succeeded") for name in STAGE_NAMES])
    monkeypatch.setattr(ri, "get_connection", db.get_connection)

    assert ri._find_resume_point() == (None, None)


def test_an_automatic_rerun_continues_the_same_run_from_the_failed_stage(ri, monkeypatch):
    db = ScriptedDB(unfinished_runs=[(5,)], stage_rows=[
        (name, "succeeded") for name in STAGE_NAMES[:10]  # everything up to and including embed_images
    ] + [("load_vector_db", "failed")])
    calls = install(ri, monkeypatch, db)

    ri.run_ingestion()

    assert calls == ["load_vector_db", "load_graph_db"]
    assert db.find("ingestion_runs (trigger)") == []                      # no NEW run was created
    assert {params[0] for _, params in db.find("ingestion_run_stages (run_id, stage)")} == {5}
    assert db.find("SET status = 'succeeded', finished_at = now() WHERE run_id")[0][1] == (5,)


def test_an_explicit_start_from_always_creates_a_new_run(ri, monkeypatch):
    db = ScriptedDB(unfinished_runs=[(5,)], stage_rows=[("load_data", "failed")])
    install(ri, monkeypatch, db)

    ri.run_ingestion(start_from="embed_text")

    assert db.find("ingestion_runs (trigger)") and db.find("SELECT run_id") == []  # never even looks for one to resume


def test_the_resume_lookup_only_considers_runs_that_did_not_succeed(ri, monkeypatch):
    db = ScriptedDB()
    monkeypatch.setattr(ri, "get_connection", db.get_connection)

    ri._find_resume_point()

    sql = db.find("SELECT run_id")[0][0]
    assert "status != 'succeeded'" in sql and "ORDER BY run_id DESC LIMIT 1" in sql
