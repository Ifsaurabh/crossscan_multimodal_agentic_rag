"""Reads the online-evaluation data back out: how is the live system doing?

Three views: `summary` (numbers over the last N days), `alerts` (which of them
look unhealthy), and `review_queue` (real questions that got a low judge score
or a thumbs down, to be reviewed and added to the golden set - this is how
production feeds back into offline evaluation).

    python src/online_report.py [days]
"""
import json
import sys

from db import SCHEMA_NAME, get_connection

DEFAULT_DAYS = 7
LOW_SCORE_THRESHOLD = 0.5

# Alert thresholds. Each needs a minimum sample so a handful of answers cannot
# raise a false alarm.
MIN_VOTES = 10
SATISFACTION_MIN = 0.6
MIN_SCORED = 5
FAITHFULNESS_MIN = 0.7
RELEVANCE_MIN = 0.7
MIN_ANSWERS = 10
P95_LATENCY_MAX_S = 60.0
BLOCK_RATE_MAX = 0.2


def _round(value, digits=3):
    return None if value is None else round(float(value), digits)


def summary(conn, days: int = DEFAULT_DAYS) -> dict:
    window = "created_at >= now() - make_interval(days => %s)"

    # Operational health, from the metadata stored with every answer.
    answers = conn.execute(
        f"""SELECT COUNT(*),
                   AVG((metadata->>'latency_s')::float),
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY (metadata->>'latency_s')::float),
                   AVG(CASE WHEN metadata->>'cache_hit' IS NULL THEN NULL
                            WHEN (metadata->>'cache_hit')::boolean THEN 1.0 ELSE 0.0 END),
                   AVG(CASE WHEN metadata->>'blocked' IS NULL THEN NULL
                            WHEN (metadata->>'blocked')::boolean THEN 1.0 ELSE 0.0 END),
                   AVG(COALESCE((metadata->>'prompt_tokens')::float, 0) + COALESCE((metadata->>'output_tokens')::float, 0))
            FROM {SCHEMA_NAME}.chat_messages
            WHERE role = 'assistant' AND {window}""",
        (days,),
    ).fetchone()

    votes = conn.execute(
        f"""SELECT COUNT(*) FILTER (WHERE rating = 1), COUNT(*) FILTER (WHERE rating = -1)
            FROM {SCHEMA_NAME}.answer_feedback WHERE {window}""",
        (days,),
    ).fetchone()

    scores = conn.execute(
        f"""SELECT COUNT(*), AVG(faithfulness), AVG(relevance)
            FROM {SCHEMA_NAME}.online_eval_scores WHERE {window}""",
        (days,),
    ).fetchone()

    up, down = int(votes[0] or 0), int(votes[1] or 0)
    return {
        "days": days,
        "answers": int(answers[0] or 0),
        "avg_latency_s": _round(answers[1], 2),
        "p95_latency_s": _round(answers[2], 2),
        "cache_hit_rate": _round(answers[3]),
        "blocked_rate": _round(answers[4]),
        "avg_tokens_per_answer": _round(answers[5], 0),
        "thumbs_up": up,
        "thumbs_down": down,
        "satisfaction": _round(up / (up + down)) if up + down else None,
        "judged_answers": int(scores[0] or 0),
        "avg_faithfulness": _round(scores[1]),
        "avg_relevance": _round(scores[2]),
    }


def alerts(report: dict) -> list:
    """Human-readable warnings for the unhealthy numbers in a summary()."""
    found = []
    votes = report["thumbs_up"] + report["thumbs_down"]
    if votes >= MIN_VOTES and report["satisfaction"] < SATISFACTION_MIN:
        found.append(f"User satisfaction is {report['satisfaction']:.0%} over {votes} votes (minimum {SATISFACTION_MIN:.0%}).")
    if report["judged_answers"] >= MIN_SCORED:
        if report["avg_faithfulness"] is not None and report["avg_faithfulness"] < FAITHFULNESS_MIN:
            found.append(f"Faithfulness is {report['avg_faithfulness']:.2f} (minimum {FAITHFULNESS_MIN}): answers may be drifting from the sources.")
        if report["avg_relevance"] is not None and report["avg_relevance"] < RELEVANCE_MIN:
            found.append(f"Relevance is {report['avg_relevance']:.2f} (minimum {RELEVANCE_MIN}).")
    if report["answers"] >= MIN_ANSWERS:
        if report["p95_latency_s"] is not None and report["p95_latency_s"] > P95_LATENCY_MAX_S:
            found.append(f"p95 latency is {report['p95_latency_s']:.0f}s (maximum {P95_LATENCY_MAX_S:.0f}s).")
        if report["blocked_rate"] is not None and report["blocked_rate"] > BLOCK_RATE_MAX:
            found.append(f"{report['blocked_rate']:.0%} of messages are blocked by guardrails (maximum {BLOCK_RATE_MAX:.0%}): a false-positive problem or an attack.")
    return found


def review_queue(conn, days: int = DEFAULT_DAYS, threshold: float = LOW_SCORE_THRESHOLD, limit: int = 20) -> list:
    """Live answers worth a human look: judged below `threshold` on either
    score, or given a thumbs down. Newest first. Each entry carries the
    (already PII-redacted) question that was asked, ready to become a golden
    set candidate once someone writes the reference answer."""
    rows = conn.execute(
        f"""SELECT a.message_id, a.created_at,
                   (SELECT u.content FROM {SCHEMA_NAME}.chat_messages u
                     WHERE u.session_id = a.session_id AND u.role = 'user' AND u.message_id < a.message_id
                     ORDER BY u.message_id DESC LIMIT 1) AS question,
                   a.content, s.faithfulness, s.relevance, s.reason, f.rating, f.comment
            FROM {SCHEMA_NAME}.chat_messages a
            LEFT JOIN {SCHEMA_NAME}.online_eval_scores s ON s.message_id = a.message_id
            LEFT JOIN {SCHEMA_NAME}.answer_feedback f ON f.message_id = a.message_id
            WHERE a.role = 'assistant'
              AND a.created_at >= now() - make_interval(days => %s)
              AND (LEAST(s.faithfulness, s.relevance) < %s OR f.rating = -1)
            ORDER BY a.created_at DESC
            LIMIT %s""",
        (days, threshold, limit),
    ).fetchall()
    return [
        {
            "message_id": r[0], "created_at": r[1], "question": r[2], "answer": r[3],
            "faithfulness": _round(r[4]), "relevance": _round(r[5]), "judge_reason": r[6],
            "thumbs": r[7], "user_comment": r[8],
        }
        for r in rows
    ]


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DAYS
    conn = get_connection()
    try:
        report = summary(conn, days)
        print(json.dumps(report, indent=2))
        for warning in alerts(report):
            print(f"ALERT: {warning}")
        queue = review_queue(conn, days)
        print(f"\n{len(queue)} answer(s) to review (low judge score or thumbs down):")
        for item in queue:
            print(f"- #{item['message_id']} faith={item['faithfulness']} rel={item['relevance']} thumbs={item['thumbs']}: {item['question']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
