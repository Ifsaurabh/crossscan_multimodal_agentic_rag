import online_report as report
from fake_db import FakeConn


def healthy(**overrides):
    base = {
        "days": 7, "answers": 100, "avg_latency_s": 8.0, "p95_latency_s": 20.0, "cache_hit_rate": 0.1,
        "blocked_rate": 0.02, "avg_tokens_per_answer": 5000, "thumbs_up": 30, "thumbs_down": 5,
        "satisfaction": 0.857, "judged_answers": 20, "avg_faithfulness": 0.9, "avg_relevance": 0.9,
    }
    base.update(overrides)
    return base


# ---------- summary ----------

def summary_conn(answers=(100, 8.123, 20.456, 0.1, 0.02, 5000.4), votes=(30, 10), scores=(20, 0.9123, 0.8)):
    return FakeConn(responses=[
        ("percentile_cont", [answers]),
        ("answer_feedback", [votes]),
        ("online_eval_scores", [scores]),
    ])


def test_summary_maps_and_rounds_the_three_queries():
    result = report.summary(summary_conn(), days=7)

    assert result["answers"] == 100 and result["avg_latency_s"] == 8.12 and result["p95_latency_s"] == 20.46
    assert result["avg_tokens_per_answer"] == 5000
    assert (result["thumbs_up"], result["thumbs_down"], result["satisfaction"]) == (30, 10, 0.75)
    assert (result["judged_answers"], result["avg_faithfulness"], result["avg_relevance"]) == (20, 0.912, 0.8)


def test_summary_of_an_empty_period_has_none_not_zero_or_a_crash():
    empty = summary_conn(answers=(0, None, None, None, None, None), votes=(0, 0), scores=(0, None, None))

    result = report.summary(empty)

    assert result["answers"] == 0 and result["avg_latency_s"] is None
    assert result["satisfaction"] is None  # no votes is not "0% satisfied"
    assert result["avg_faithfulness"] is None


def test_summary_passes_the_window_to_every_query():
    conn = summary_conn()

    report.summary(conn, days=30)

    assert all(params == (30,) for _, params in conn.executed) and len(conn.executed) == 3


# ---------- alerts ----------

def test_a_healthy_system_raises_no_alerts():
    assert report.alerts(healthy()) == []


def test_low_satisfaction_alerts_only_with_enough_votes():
    assert report.alerts(healthy(thumbs_up=1, thumbs_down=5, satisfaction=1 / 6)) == []  # 6 votes: too few to judge
    found = report.alerts(healthy(thumbs_up=4, thumbs_down=10, satisfaction=4 / 14))

    assert len(found) == 1 and "satisfaction" in found[0]


def test_low_faithfulness_and_relevance_alert_only_with_enough_judged_answers():
    assert report.alerts(healthy(judged_answers=3, avg_faithfulness=0.1, avg_relevance=0.1)) == []
    found = report.alerts(healthy(judged_answers=10, avg_faithfulness=0.5, avg_relevance=0.6))

    assert any("Faithfulness" in f for f in found) and any("Relevance" in f for f in found)


def test_slow_and_over_blocking_systems_alert_only_with_enough_traffic():
    assert report.alerts(healthy(answers=3, p95_latency_s=999, blocked_rate=0.9)) == []
    found = report.alerts(healthy(answers=50, p95_latency_s=90.0, blocked_rate=0.4))

    assert any("latency" in f for f in found) and any("blocked" in f for f in found)


def test_missing_scores_never_crash_the_alert_check():
    assert report.alerts(healthy(judged_answers=10, avg_faithfulness=None, avg_relevance=None)) == []


# ---------- review queue ----------

def test_review_queue_maps_rows_and_passes_window_threshold_and_limit():
    row = (9, "t", "what accuracy?", "It was 94%.", 0.31234, 0.9, "unsupported claim", -1, "wrong number")
    conn = FakeConn(responses=[("LEAST(s.faithfulness, s.relevance)", [row])])

    queue = report.review_queue(conn, days=14, threshold=0.4, limit=5)

    assert conn.executed[0][1] == (14, 0.4, 5)
    assert queue == [{
        "message_id": 9, "created_at": "t", "question": "what accuracy?", "answer": "It was 94%.",
        "faithfulness": 0.312, "relevance": 0.9, "judge_reason": "unsupported claim",
        "thumbs": -1, "user_comment": "wrong number",
    }]


def test_review_queue_selects_low_scores_or_thumbs_down_from_assistant_answers_only():
    conn = FakeConn()

    report.review_queue(conn)

    sql = conn.executed[0][0]
    assert "a.role = 'assistant'" in sql and "f.rating = -1" in sql and "< %s" in sql
