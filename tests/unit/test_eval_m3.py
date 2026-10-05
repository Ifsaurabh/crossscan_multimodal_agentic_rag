"""The evaluation's new metrics and follow-up questions (golden_set_m3.jsonl), where a run is stored, and which models answered."""
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from evaluation import experiment_tracking as et
from evaluation import run_evaluation as re_
from shared import llm_connection as lc


def make_result(chunks=None, **extra):
    if chunks is None:
        chunks = [{"source_pdf": "a.pdf", "page_start": 3, "page_end": 5, "parent_text": "parent A", "text": "child A"}]
    result = {"final_answer": "98% accuracy [a.pdf, p.4]", "blocked": False, "cache_hit": False, "guardrail_flags": [],
              "sub_queries": [{"chunks": chunks, "images": [], "tables": []}]}
    result.update(extra)
    return result


def with_media(images=(), tables=()):
    result = make_result()
    result["sub_queries"][0]["images"], result["sub_queries"][0]["tables"] = list(images), list(tables)
    return result


RECORD = {"input": "What accuracy?", "expected_output": "98%", "context": ["ctx"], "source_pdf": "a.pdf", "domain": "x"}


# ---------- metrics ----------

def test_a_golden_set_written_with_plain_file_names_matches_the_domain_prefixed_names_the_pipeline_stores():
    chunks = [{"source_pdf": "lung-cancer/a.pdf", "page_start": 3, "parent_text": "p", "text": "c"}]
    assert re_.deterministic_metrics({"source_pdf": "a.pdf"}, make_result(chunks=chunks), 0.1)["source_hit"] == 1.0
    assert re_.deterministic_metrics({"source_pdf": "lung-cancer/a.pdf"}, make_result(), 0.1)["source_hit"] == 1.0
    assert re_.deterministic_metrics({"source_pdf": "b.pdf"}, make_result(chunks=chunks), 0.1)["source_hit"] == 0.0
    assert re_.deterministic_metrics({}, make_result(), 0.1)["source_hit"] == 0.0  # no expected source: never a hit


def test_the_new_metrics_are_none_for_a_question_that_does_not_expect_them():
    metrics = re_.deterministic_metrics(RECORD, with_media([{"image_file": "f.png"}], [{"source_pdf": "a.pdf"}]), 0.1)
    assert (metrics["image_hit"], metrics["table_hit"], metrics["listing_recall"]) == (None, None, None)


def test_a_question_that_expects_a_figure_scores_whether_one_was_shown():
    record = {**RECORD, "expects_image": True}
    assert re_.deterministic_metrics(record, with_media([{"image_file": "f.png"}]), 0.1)["image_hit"] == 1.0
    assert re_.deterministic_metrics(record, with_media(), 0.1)["image_hit"] == 0.0


def test_a_question_that_expects_a_table_scores_whether_a_table_of_the_right_paper_was_used():
    record = {**RECORD, "source_pdf": "a.pdf", "expects_table": True}
    right = with_media(tables=[{"source_pdf": "lung-cancer/a.pdf", "table_id": "t"}])
    wrong = with_media(tables=[{"source_pdf": "land-cover/b.pdf", "table_id": "t"}])
    assert re_.deterministic_metrics(record, right, 0.1)["table_hit"] == 1.0
    assert re_.deterministic_metrics(record, wrong, 0.1)["table_hit"] == 0.0
    assert re_.deterministic_metrics(record, with_media(), 0.1)["table_hit"] == 0.0


def test_a_listing_question_scores_the_share_of_the_expected_papers_it_found():
    chunks = [{"source_pdf": "land-cover/a.pdf", "page_start": 1, "text": "c"}, {"source_pdf": "b.pdf", "page_start": 1, "text": "c"}]
    record = {**RECORD, "expected_sources": ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]}
    assert re_.deterministic_metrics(record, make_result(chunks=chunks), 0.1)["listing_recall"] == 0.5


def test_the_new_metrics_are_in_the_list_of_deterministic_metrics():
    assert {"image_hit", "table_hit", "listing_recall"} <= set(re_.DETERMINISTIC_METRICS)


# ---------- follow-ups and models ----------

def test_a_follow_up_is_given_its_earlier_turns_as_the_planner_receives_them():
    seen = {}

    def invoke(query, history=None):
        seen.update(query=query, history=history)
        return make_result()

    record = {**RECORD, "input": "And its recall?", "history": [
        {"role": "user", "content": "Which F1?"},
        {"role": "assistant", "content": "92", "metadata": {"sources": [{"source_pdf": "land-cover/a.pdf", "page": 10}]}}]}

    re_.evaluate_record(invoke, record)

    assert seen["query"] == "And its recall?"
    assert seen["history"]["recent_turns"][1]["citations"] == [{"paper": "land-cover/a.pdf", "pages": [10]}]


def test_a_question_without_history_is_invoked_as_before():
    row = re_.evaluate_record(lambda q: make_result(), RECORD)  # an invoke that takes no history argument still works
    assert row["input"] == RECORD["input"]


def test_the_row_keeps_which_models_answered():
    used = [{"task": "answer", "tier": "fast", "provider": "gemini", "model": "gemini-3.5-flash-lite", "fallback": True}]
    row = re_.evaluate_record(lambda q: make_result(models_used=used), RECORD)
    assert row["models_used"] == used


def test_the_extra_golden_file_is_valid_and_its_extra_fields_are_known():
    path = Path(__file__).resolve().parents[2] / "data" / "golden_set_m3.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(records) >= 8
    allowed = {"input", "expected_output", "context", "source_pdf", "domain", "expects_table", "expects_image",
               "expected_sources", "history"}
    for record in records:
        assert set(record) <= allowed and record["input"] and record["expected_output"]
        for message in record.get("history", []):
            assert message["role"] in ("user", "assistant") and message["content"]
    assert any(r.get("history") for r in records) and any(r.get("expects_image") for r in records)
    assert any(r.get("expected_sources") for r in records) and any(r.get("expects_table") for r in records)


# ---------- where a run is stored, and what it records ----------

def test_a_run_goes_to_the_store_named_by_the_environment_with_its_files_in_the_artifact_root(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient

    store, artifacts = tmp_path / "store", tmp_path / "artifacts"
    store.mkdir()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store.as_uri())
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", artifacts.as_uri())
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    run_id = et.log_run("shared-store", params={"n": 1}, metrics={"source_hit": 1.0}, results_rows=[{"input": "q"}])

    client = MlflowClient(tracking_uri=store.as_uri())
    run = client.get_run(run_id)
    assert run.info.artifact_uri.startswith(artifacts.as_uri())  # the files went to the artifact root, not next to the store
    assert [a.path for a in client.list_artifacts(run_id, "results")]


def test_the_commit_comes_from_ci_when_there_is_one(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    assert et.git_commit() == "abc123"


def test_the_corpus_version_changes_when_a_document_changes(monkeypatch):
    def with_rows(rows):
        class Conn:
            def execute(self, sql, params=None):
                return type("C", (), {"fetchall": lambda self: rows})()

        @contextmanager
        def borrow():
            yield Conn()

        monkeypatch.setattr("shared.db.connection", borrow)

    with_rows([("lung-cancer/a.pdf", "h1"), ("land-cover/b.pdf", "h2")])
    first = et.corpus_version()
    with_rows([("lung-cancer/a.pdf", "h1-changed"), ("land-cover/b.pdf", "h2")])
    second = et.corpus_version()

    assert first[1] == second[1] == 2 and first[0] != second[0] and len(first[0]) == 12


def unreachable(monkeypatch):
    def down():
        raise RuntimeError("database unreachable")

    monkeypatch.setattr("shared.db.connection", down)


def test_an_unreadable_database_gives_an_unknown_corpus_not_an_error(monkeypatch):
    unreachable(monkeypatch)
    assert et.corpus_version() == ("unknown", 0)


def test_a_run_records_the_commit_and_the_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    unreachable(monkeypatch)
    golden = tmp_path / "g.jsonl"
    golden.write_bytes(b"x")

    versions = et.collect_versions(golden)

    assert versions["git_commit"] == "deadbeef" and versions["corpus_version"] == "unknown" and versions["corpus_documents"] == 0


# ---------- which model answered ----------

@pytest.fixture
def one_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setitem(lc._ADAPTERS, "gemini", lambda config, model, system, user, client=None:
                        lc.LLMResult(text="ok", provider="gemini", model=model))


def test_the_models_that_answered_are_recorded_when_recording_is_on(one_gemini):
    lc.start_recording_models()

    lc.generate("s", "u", task="query_planner")
    lc.generate("s", "u", tier="fast")

    used = lc.recorded_models()
    assert [u["task"] for u in used] == ["query_planner", None] and used[0]["provider"] == "gemini"
    assert used[0]["fallback"] is False and used[0]["model"]


def test_nothing_is_recorded_unless_recording_was_started(one_gemini):
    lc._models_used.set(None)
    lc.generate("s", "u", tier="fast")
    assert lc.recorded_models() == []


def test_a_new_question_starts_with_an_empty_list(one_gemini):
    lc.start_recording_models()
    lc.generate("s", "u", tier="fast")
    lc.start_recording_models()
    assert lc.recorded_models() == []
