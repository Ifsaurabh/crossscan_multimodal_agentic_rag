"""Online evaluation: quality scoring of a SAMPLE of real answers, in production.

Offline evaluation (run_evaluation.py) scores a fixed golden set before a
release. This module watches live traffic, where there is no known-good answer,
so it uses reference-free DeepEval metrics: is the answer supported by the context
that was retrieved (FaithfulnessMetric), and does it address the question (AnswerRelevancyMetric)?
The metrics run on the "evaluation" tier models (shared/deepeval_gemini_model.py), like the offline evaluation.

Design rules:
- Sampled (ONLINE_EVAL_SAMPLE_RATE, default 10%; 0 turns it off) because every
  judged answer costs several evaluation-tier model calls (DeepEval splits the answer into claims and checks each).
- Runs in a background thread AFTER the reply is stored: the user never waits
  for it, is never charged for it, and a failure here can never fail a turn.
- The question, context and answer are untrusted text: they are length-limited, and DeepEval's own prompts treat them as
  material to check. A judged score is a signal for the health report, never something an answer can gate on.
"""
import os
import random
import threading

from shared import llm_connection
from shared.db import SCHEMA_NAME, connection
from retrieval.result_context import collect_contexts

DEFAULT_SAMPLE_RATE = 0.1
MAX_CONTEXT_CHARS_EACH = 1500
MAX_CONTEXT_CHARS_TOTAL = 6000
MAX_ANSWER_CHARS = 3000
MAX_REASON_CHARS = 300


def sample_rate() -> float:
    """Share of answers to judge, 0.0-1.0. A bad value falls back to the default."""
    raw = os.environ.get("ONLINE_EVAL_SAMPLE_RATE")
    if raw in (None, ""):
        return DEFAULT_SAMPLE_RATE
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return DEFAULT_SAMPLE_RATE


def should_sample(rate: float = None, rng=random.random) -> bool:
    rate = sample_rate() if rate is None else rate
    return rate > 0 and rng() < rate


def limit_contexts(contexts: list) -> list:
    """The retrieved passages, each cut short and only as many as fit the total, so one judged answer stays cheap."""
    kept, used = [], 0
    for text in contexts:
        text = text[:MAX_CONTEXT_CHARS_EACH]
        if used + len(text) > MAX_CONTEXT_CHARS_TOTAL:
            break
        kept.append(text)
        used += len(text)
    return kept


def _score(value, name: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not a number: {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} out of range: {value}")
    return float(value)


def judge(query: str, answer: str, contexts: list, model=None, faithfulness_metric=None, relevancy_metric=None) -> dict:
    """DeepEval's faithfulness and answer-relevancy scores for one answer, from the evaluation-tier models. Returns
    {"faithfulness", "relevance", "reason", "judge_model"}. Faithfulness is None when nothing was retrieved (a
    general-knowledge answer has nothing to be faithful to). Raises when a score is missing or out of range, so a
    failed judgement is dropped, never stored as a bogus number."""
    # Imported here: DeepEval is slow to import and only a judged answer needs it.
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    from shared.deepeval_gemini_model import GeminiDeepEvalModel

    if model is None:
        model = GeminiDeepEvalModel()
    contexts = limit_contexts(contexts)
    case = LLMTestCase(input=query, actual_output=answer[:MAX_ANSWER_CHARS], retrieval_context=contexts or None)
    llm_connection.start_recording_models()

    relevancy = relevancy_metric or AnswerRelevancyMetric(model=model, async_mode=False, verbose_mode=False)
    relevancy.measure(case)
    reasons = [getattr(relevancy, "reason", None)]
    faithfulness = None
    if contexts:
        metric = faithfulness_metric or FaithfulnessMetric(model=model, async_mode=False, verbose_mode=False)
        metric.measure(case)
        faithfulness = _score(metric.score, "faithfulness")
        reasons.append(getattr(metric, "reason", None))

    models = []
    for used in llm_connection.recorded_models():
        if used["model"] not in models:
            models.append(used["model"])
    return {
        "faithfulness": faithfulness,
        "relevance": _score(relevancy.score, "relevance"),
        "reason": " ".join(r for r in reasons if r)[:MAX_REASON_CHARS],
        "judge_model": ", ".join(models) or None,
    }


def record_score(conn, message_id: int, scores: dict) -> None:
    conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.online_eval_scores (message_id, faithfulness, relevance, judge_model, reason)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (message_id) DO UPDATE SET
                faithfulness = EXCLUDED.faithfulness, relevance = EXCLUDED.relevance,
                judge_model = EXCLUDED.judge_model, reason = EXCLUDED.reason, created_at = now()""",
        (message_id, scores["faithfulness"], scores["relevance"], scores.get("judge_model"), scores["reason"]),
    )
    conn.commit()


def maybe_score(
    message_id, query: str, answer: str, result: dict, blocked: bool = False,
    background: bool = True, rate: float = None, rng=random.random,
    judge_fn=None, conn_factory=None,
) -> bool:
    """Judges this answer if it is sampled. Returns True when it was picked
    (whether or not the judge then succeeded). Never raises."""
    if blocked or not answer or message_id is None:
        return False
    if not should_sample(rate, rng):
        return False

    contexts = collect_contexts([c for sq in result.get("sub_queries", []) for c in sq.get("chunks", [])])
    resolved = " | ".join(sq.get("sub_query", "") for sq in result.get("sub_queries", []) if sq.get("sub_query"))
    judged_query = resolved or query
    judge_fn = judge_fn or judge

    def run():
        try:
            scores = judge_fn(judged_query, answer, contexts)
            if conn_factory is None:
                with connection() as conn:
                    record_score(conn, message_id, scores)
            else:
                conn = conn_factory()
                try:
                    record_score(conn, message_id, scores)
                finally:
                    conn.close()
        except Exception as e:
            print(f"   (online evaluation skipped: {e})")

    if background:
        threading.Thread(target=run, name="online-eval", daemon=True).start()
    else:
        run()
    return True
