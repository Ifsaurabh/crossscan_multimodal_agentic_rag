import threading

import pytest

from retrieval import online_eval
from fake_db import FakeConn


RESULT = {"sub_queries": [{"sub_query": "CNN accuracy?", "chunks": [{"parent_text": "The CNN reached 94%."}]}]}


# ---------- sampling ----------

def test_sample_rate_defaults_to_ten_percent_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv("ONLINE_EVAL_SAMPLE_RATE", raising=False)
    assert online_eval.sample_rate() == 0.1
    monkeypatch.setenv("ONLINE_EVAL_SAMPLE_RATE", "0.25")
    assert online_eval.sample_rate() == 0.25


@pytest.mark.parametrize("raw,expected", [("5", 1.0), ("-1", 0.0), ("lots", 0.1), ("", 0.1)])
def test_sample_rate_is_clamped_and_bad_values_fall_back(monkeypatch, raw, expected):
    monkeypatch.setenv("ONLINE_EVAL_SAMPLE_RATE", raw)

    assert online_eval.sample_rate() == expected


def test_should_sample_follows_the_rate():
    assert online_eval.should_sample(0.5, rng=lambda: 0.49) is True
    assert online_eval.should_sample(0.5, rng=lambda: 0.5) is False
    assert online_eval.should_sample(0.0, rng=lambda: 0.0) is False  # 0 switches it off entirely
    assert online_eval.should_sample(1.0, rng=lambda: 0.999) is True


def test_sampling_rate_is_statistically_about_right():
    import random

    rng = random.Random(7).random
    picked = sum(online_eval.should_sample(0.1, rng=rng) for _ in range(5000))

    assert 400 < picked < 600  # ~500 expected


# ---------- the judge: DeepEval's metrics on the evaluation tier ----------

class FakeMetric:
    """Stands in for a DeepEval metric: sets a score and a reason, and notes the model that 'answered'."""

    def __init__(self, score, reason="r", model="gemini-3.5-flash"):
        self.score, self.reason, self._model, self.cases = score, reason, model, []

    def measure(self, case):
        from shared import llm_connection

        self.cases.append(case)
        llm_connection._note_model("deepeval_judge", "evaluation", "gemini", self._model, False)


def judged(query="q?", answer="an answer", contexts=("some context",), faithfulness=0.9, relevancy=0.8, **kw):
    f, r = FakeMetric(faithfulness, "supported"), FakeMetric(relevancy, "on topic")
    scores = online_eval.judge(query, answer, list(contexts), model=object(), faithfulness_metric=f, relevancy_metric=r, **kw)
    return scores, f, r


def test_the_judge_reports_both_deepeval_scores_the_reasons_and_the_evaluation_tier_model():
    scores, _, _ = judged()

    assert scores == {"faithfulness": 0.9, "relevance": 0.8, "reason": "on topic supported", "judge_model": "gemini-3.5-flash"}


def test_the_metrics_get_the_question_the_answer_and_the_retrieved_passages():
    _, faithfulness, relevancy = judged(query="CNN accuracy?", answer="94%", contexts=["The CNN reached 94%."])

    case = faithfulness.cases[0]
    assert (case.input, case.actual_output, case.retrieval_context) == ("CNN accuracy?", "94%", ["The CNN reached 94%."])
    assert relevancy.cases[0] is not None


def test_no_context_means_faithfulness_is_not_applicable_and_is_not_measured():
    scores, faithfulness, _ = judged(contexts=[], faithfulness=0.0)

    assert scores["faithfulness"] is None and scores["relevance"] == 0.8 and faithfulness.cases == []


def test_the_judge_records_every_model_that_answered_once():
    class Mixed(FakeMetric):
        def measure(self, case):
            super().measure(case)
            from shared import llm_connection
            llm_connection._note_model("deepeval_judge", "evaluation", "gemini", "gemini-3.5-flash-lite", True)

    f, r = Mixed(0.9), Mixed(0.8)
    scores = online_eval.judge("q", "a", ["c"], model=object(), faithfulness_metric=f, relevancy_metric=r)

    assert scores["judge_model"] == "gemini-3.5-flash, gemini-3.5-flash-lite"


@pytest.mark.parametrize("bad", [None, 1.5, -0.1, "high", True])
def test_a_missing_or_out_of_range_score_is_rejected_not_stored(bad):
    with pytest.raises(ValueError):
        judged(relevancy=bad)
    with pytest.raises(ValueError):
        judged(faithfulness=bad)


def test_the_reason_is_length_limited():
    f, r = FakeMetric(0.5, "x" * 1000), FakeMetric(0.5, "y" * 1000)

    scores = online_eval.judge("q", "a", ["c"], model=object(), faithfulness_metric=f, relevancy_metric=r)

    assert len(scores["reason"]) == online_eval.MAX_REASON_CHARS


def test_prompt_size_is_bounded():
    huge = ["Ж" * 10_000] * 20

    scores, faithfulness, _ = judged(answer="Ω" * 50_000, contexts=huge)

    case = faithfulness.cases[0]
    assert len(case.actual_output) == online_eval.MAX_ANSWER_CHARS
    assert sum(len(c) for c in case.retrieval_context) <= online_eval.MAX_CONTEXT_CHARS_TOTAL
    assert all(len(c) <= online_eval.MAX_CONTEXT_CHARS_EACH for c in case.retrieval_context)


def test_the_online_judge_runs_deepeval_through_the_evaluation_tier():
    from shared import llm_connection

    assert llm_connection.TASK_TIERS["deepeval_judge"] == "evaluation"


# ---------- storing ----------

def test_record_score_upserts_by_message():
    conn = FakeConn()

    online_eval.record_score(conn, 42, {"faithfulness": 0.9, "relevance": 0.8, "judge_model": "m", "reason": "r"})

    sql, params = conn.find("online_eval_scores")[0]
    assert "ON CONFLICT (message_id) DO UPDATE" in sql
    assert params == (42, 0.9, 0.8, "m", "r")
    assert conn.commits == 1


# ---------- the sampled, background hook ----------

def run(monkeypatch=None, **kwargs):
    conn, recorded, judged = FakeConn(), [], []

    def judge_fn(query, answer, contexts):
        judged.append((query, answer, contexts))
        return {"faithfulness": 0.9, "relevance": 0.8, "judge_model": "m", "reason": "r"}

    args = dict(
        message_id=5, query="raw query", answer="an answer", result=RESULT, blocked=False,
        background=False, rate=1.0, judge_fn=judge_fn, conn_factory=lambda: conn,
    )
    args.update(kwargs)
    picked = online_eval.maybe_score(**args)
    return picked, conn, judged


def test_a_sampled_answer_is_judged_and_stored_using_the_resolved_question():
    picked, conn, judged = run()

    assert picked is True
    assert judged == [("CNN accuracy?", "an answer", ["The CNN reached 94%."])]  # resolved sub-query, parent text
    assert conn.find("INSERT INTO") and conn.closed is True


def test_the_raw_query_is_used_when_there_are_no_sub_queries():
    _, _, judged = run(result={})

    assert judged[0][0] == "raw query"


@pytest.mark.parametrize("override", [{"blocked": True}, {"answer": ""}, {"message_id": None}])
def test_blocked_empty_or_unsaved_answers_are_never_judged(override):
    picked, conn, judged = run(**override)

    assert picked is False and judged == [] and conn.executed == []


def test_an_answer_that_is_not_sampled_costs_nothing():
    picked, conn, judged = run(rate=0.0)

    assert picked is False and judged == []


def test_a_judge_failure_is_swallowed_and_nothing_is_stored(capsys):
    def broken(query, answer, contexts):
        raise RuntimeError("quota exhausted")

    picked, conn, _ = run(judge_fn=broken)

    assert picked is True  # it was sampled; the failure is only logged
    assert conn.find("INSERT INTO") == []
    assert "online evaluation skipped" in capsys.readouterr().out


def test_a_database_failure_is_swallowed_and_the_connection_still_closes():
    class Broken(FakeConn):
        def execute(self, sql, params=None):
            raise RuntimeError("db down")

    conn = Broken()

    picked, _, _ = run(conn_factory=lambda: conn)

    assert picked is True and conn.closed is True


def test_background_mode_does_not_block_the_caller():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow_judge(query, answer, contexts):
        started.set()
        release.wait(timeout=5)
        finished.set()
        return {"faithfulness": 1.0, "relevance": 1.0, "judge_model": "m", "reason": ""}

    picked = online_eval.maybe_score(
        5, "q", "a", RESULT, background=True, rate=1.0, judge_fn=slow_judge, conn_factory=FakeConn,
    )

    assert picked is True and not finished.is_set()  # returned while the judge is still running
    assert started.wait(timeout=5)
    release.set()
    assert finished.wait(timeout=5)
