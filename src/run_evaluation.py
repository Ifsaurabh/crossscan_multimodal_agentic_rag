import argparse
import asyncio
import json
import time
from pathlib import Path

import experiment_tracking
import llm_connection
import output_guardrail
import usage_tracker

GOLDEN_SET_PATH = Path("data") / "golden_set.jsonl"

DETERMINISTIC_METRICS = [
    "source_hit",
    "citation_verified_rate",
    "is_grounded",
    "guardrail_flag_count",
    "answer_chars",
    "latency_s",
]


def load_golden_set(path=GOLDEN_SET_PATH, limit=None) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[:limit] if limit else records


def collect_chunks(result: dict) -> list:
    """Every chunk retrieved across all sub-queries of a graph run."""
    chunks = []
    for sq in result.get("sub_queries", []):
        chunks.extend(sq.get("chunks", []))
    return chunks


def collect_contexts(chunks: list) -> list:
    """Retrieved context strings (parent text when present), de-duplicated in
    order - the shape RAGAS/DeepEval expect for retrieval_context."""
    seen, contexts = set(), []
    for c in chunks:
        text = c.get("parent_text") or c.get("text") or ""
        if text and text not in seen:
            seen.add(text)
            contexts.append(text)
    return contexts


def deterministic_metrics(record: dict, result: dict, latency_s: float) -> dict:
    """Metrics that need no LLM judge - free, instant, and safe to run on
    every query at any scale."""
    chunks = collect_chunks(result)
    answer = result.get("final_answer") or ""
    sources = {c.get("source_pdf") for c in chunks}

    check = output_guardrail.check_output(answer, chunks)
    verified = len(check["verified_citations"])
    total = verified + len(check["unverified_citations"])

    return {
        "source_hit": 1.0 if record.get("source_pdf") in sources else 0.0,
        "citation_verified_rate": (verified / total) if total else None,
        "is_grounded": 1.0 if check["is_grounded"] else 0.0,
        "guardrail_flag_count": float(len(result.get("guardrail_flags", []))),
        "answer_chars": float(len(answer)),
        "latency_s": latency_s,
    }


def evaluate_record(invoke, record: dict) -> dict:
    """Runs one golden question through the pipeline via `invoke(query)` and
    returns a flat result row."""
    start = time.time()
    result = invoke(record["input"])
    latency = round(time.time() - start, 3)

    chunks = collect_chunks(result)
    return {
        "input": record["input"],
        "expected_output": record["expected_output"],
        "expected_source": record.get("source_pdf"),
        "domain": record.get("domain"),
        "answer": result.get("final_answer"),
        "blocked": bool(result.get("blocked")),
        "cache_hit": bool(result.get("cache_hit")),
        "contexts": collect_contexts(chunks),
        "metrics": deterministic_metrics(record, result, latency),
    }


def run_pipeline(invoke, records: list) -> list:
    rows = []
    for i, record in enumerate(records, 1):
        row = evaluate_record(invoke, record)
        print(f"  [{i}/{len(records)}] source_hit={row['metrics']['source_hit']:.0f} "
              f"latency={row['metrics']['latency_s']}s  {record['input'][:60]}")
        rows.append(row)
    return rows


def aggregate(rows: list) -> dict:
    """Mean of every metric across rows, skipping rows where it's None."""
    names = {name for row in rows for name in row["metrics"]}
    out = {}
    for name in sorted(names):
        values = [row["metrics"][name] for row in rows if row["metrics"].get(name) is not None]
        if values:
            out[name] = sum(values) / len(values)
    return out


def score_deepeval(rows: list, model=None) -> None:
    """LLM-judge metrics via DeepEval, written into each row's metrics.
    Costs several Gemini calls per row - mind the free-tier daily quota."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    if model is None:
        from deepeval_gemini_model import GeminiDeepEvalModel

        model = GeminiDeepEvalModel()

    metric_classes = {
        "deepeval_faithfulness": FaithfulnessMetric,
        "deepeval_answer_relevancy": AnswerRelevancyMetric,
        "deepeval_contextual_precision": ContextualPrecisionMetric,
        "deepeval_contextual_recall": ContextualRecallMetric,
    }
    for row in rows:
        if not row["contexts"] or not row["answer"]:
            continue
        case = LLMTestCase(
            input=row["input"],
            actual_output=row["answer"],
            expected_output=row["expected_output"],
            retrieval_context=row["contexts"],
        )
        for name, metric_class in metric_classes.items():
            metric = metric_class(model=model, async_mode=False, verbose_mode=False)
            metric.measure(case)
            row["metrics"][name] = metric.score


def _ragas_llm_for(spec: dict):
    """A native RAGAS LLM for one evaluation-tier model. RAGAS needs a
    provider client (not a text-in/text-out function), so it cannot go through
    llm_connection.generate(); this builds the client for the tier's model."""
    import os

    from ragas.llms import llm_factory

    provider, model = spec["provider"], spec["model"]
    api_key = os.environ[spec["config"]["api_key_env"]]

    if provider == "gemini":
        from google import genai

        return llm_factory(model, provider="google", client=genai.Client(api_key=api_key))
    if provider == "openai":
        import openai

        return llm_factory(model, provider="openai", client=openai.OpenAI(api_key=api_key))
    if provider == "anthropic":
        import anthropic

        return llm_factory(model, provider="anthropic", client=anthropic.Anthropic(api_key=api_key))
    raise ValueError(f"No RAGAS client for provider '{provider}'.")


async def _ragas_score_row(row: dict, faithfulness, precision, recall) -> None:
    args = {"user_input": row["input"], "retrieved_contexts": row["contexts"]}
    row["metrics"]["ragas_faithfulness"] = (await faithfulness.ascore(response=row["answer"], **args)).value
    row["metrics"]["ragas_context_precision"] = (await precision.ascore(reference=row["expected_output"], **args)).value
    row["metrics"]["ragas_context_recall"] = (await recall.ascore(reference=row["expected_output"], **args)).value


def score_ragas(rows: list, llm=None) -> None:
    """LLM-judge metrics via RAGAS (faithfulness, context precision/recall),
    written into each row's metrics. Judges run on the "evaluation" tier:
    the first ready model scores rows, and if it fails (quota, outage) the
    tier's next model takes over from that row on. A tier never borrows
    another tier's models. `llm` injects a ready RAGAS LLM (tests)."""
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

    if llm is not None:
        candidates = [{"provider": "injected", "model": "injected", "llm": llm}]
    else:
        candidates = [{**spec, "llm": None} for spec in llm_connection.ready_models("evaluation")]

    position = 0
    attempts = []
    for row in rows:
        if not row["contexts"] or not row["answer"]:
            continue
        while True:
            if position >= len(candidates):
                raise llm_connection.AllProvidersFailed(attempts, "evaluation")
            spec = candidates[position]
            try:
                if "metrics" not in spec:
                    judge = spec["llm"] or _ragas_llm_for(spec)
                    spec["metrics"] = (Faithfulness(llm=judge), ContextPrecision(llm=judge), ContextRecall(llm=judge))
                asyncio.run(_ragas_score_row(row, *spec["metrics"]))
                break
            except Exception as e:
                kind = "other"
                if spec["provider"] != "injected":
                    kind = llm_connection.record_failure(spec["provider"], spec["model"], e)
                attempts.append({
                    "provider": spec["provider"], "model": spec["model"], "status": "failed",
                    "kind": kind, "detail": f"{type(e).__name__}: {str(e)[:200]}",
                })
                print(f"   (RAGAS judge {spec['provider']}:{spec['model']} failed [{kind}], trying the next evaluation model...)")
                position += 1


def main():
    parser = argparse.ArgumentParser(description="Run the golden set through the RAG pipeline and log an MLflow run.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N golden examples")
    parser.add_argument(
        "--llm-metrics", choices=["none", "deepeval", "ragas", "both"], default="none",
        help="LLM-judge metrics (uses Gemini quota); deterministic metrics always run",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()

    from retrieval_graph import build_graph
    from run_query import invoke_graph

    graph = build_graph()
    records = load_golden_set(limit=args.limit)
    usage_before = usage_tracker.snapshot()
    print(f"Running {len(records)} golden example(s) through the pipeline...")
    rows = run_pipeline(lambda q: invoke_graph(graph, q, use_cache=False), records)

    if args.llm_metrics in ("deepeval", "both"):
        print("Scoring with DeepEval...")
        score_deepeval(rows)
    if args.llm_metrics in ("ragas", "both"):
        print("Scoring with RAGAS...")
        score_ragas(rows)

    summary = aggregate(rows)
    for name, value in usage_tracker.delta(usage_before, usage_tracker.snapshot()).items():
        summary[f"usage_{name}"] = value
    print("\nAggregate metrics:")
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")

    if not args.no_mlflow:
        params = experiment_tracking.collect_versions(GOLDEN_SET_PATH)
        params["n_examples"] = len(rows)
        params["llm_metrics"] = args.llm_metrics
        run_id = experiment_tracking.log_run("golden-set-eval", params, summary, rows)
        print(f"\nLogged MLflow run {run_id} (view: mlflow ui --backend-store-uri {experiment_tracking.MLRUNS_DIR.as_uri()})")


if __name__ == "__main__":
    main()
