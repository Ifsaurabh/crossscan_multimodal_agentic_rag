import argparse
import asyncio
import json
import time
from pathlib import Path

from evaluation import experiment_tracking
from shared import llm_connection
from retrieval import output_guardrail
from retrieval import memory
from retrieval.result_context import collect_chunks, collect_contexts  # noqa: F401  (used below, and re-exported)
from shared import usage_tracker

GOLDEN_SET_PATH = Path("data") / "golden_set.jsonl"

DETERMINISTIC_METRICS = [
    "source_hit",
    "citation_verified_rate",
    "is_grounded",
    "guardrail_flag_count",
    "answer_chars",
    "latency_s",
    "image_hit",
    "table_hit",
    "listing_recall",
    "retrieval_s",
    "db_s",
    "embed_s",
    "rerank_s",
]


def load_golden_set(path=GOLDEN_SET_PATH, limit=None) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[:limit] if limit else records


def _file(name) -> str:
    """The file name of a document name: `lung-cancer/paper.pdf` and `paper.pdf` are the same file, so a golden set written
    with plain file names still matches the `<domain>/<file>` names the pipeline stores."""
    return (name or "").rsplit("/", 1)[-1]


def deterministic_metrics(record: dict, result: dict, latency_s: float) -> dict:
    """Metrics that need no LLM judge - free, instant, and safe to run on
    every query at any scale."""
    chunks = collect_chunks(result)
    answer = result.get("final_answer") or ""
    sources = {_file(c.get("source_pdf")) for c in chunks if c.get("source_pdf")}
    sub_queries = result.get("sub_queries", [])
    images = [i for sq in sub_queries for i in sq.get("images", [])]
    tables = [t for sq in sub_queries for t in sq.get("tables", [])]
    expected_file = _file(record.get("source_pdf"))
    expected_sources = {_file(s) for s in record.get("expected_sources") or []}

    check = output_guardrail.check_output(answer, chunks)
    verified = len(check["verified_citations"])
    total = verified + len(check["unverified_citations"])

    timings = result.get("timings") or {}
    return {
        "source_hit": 1.0 if expected_file and expected_file in sources else 0.0,
        "citation_verified_rate": (verified / total) if total else None,
        "is_grounded": 1.0 if check["is_grounded"] else 0.0,
        "guardrail_flag_count": float(len(result.get("guardrail_flags", []))),
        "answer_chars": float(len(answer)),
        "latency_s": latency_s,
        # only for the questions that expect it (golden_set_m3.jsonl): a figure shown, the right table used, the share of
        # the expected papers a listing question found; None for every other question
        "image_hit": (1.0 if images else 0.0) if record.get("expects_image") else None,
        "table_hit": (1.0 if any(_file(t.get("source_pdf")) == expected_file for t in tables) else 0.0)
        if record.get("expects_table") else None,
        "listing_recall": (len(expected_sources & sources) / len(expected_sources)) if expected_sources else None,
        # where the time went (retrieval, the database, the query embedding, the reranker); None when not measured
        "retrieval_s": timings.get("retrieval_s"),
        "db_s": timings.get("db_s"),
        "embed_s": timings.get("embed_s"),
        "rerank_s": timings.get("rerank_s"),
    }


def evaluate_record(invoke, record: dict) -> dict:
    """Runs one golden question through the pipeline via `invoke(query)` and
    returns a flat result row."""
    start = time.time()
    if record.get("history"):
        # a follow-up: the earlier turns ({"role", "content", "sources"?}) are given to the planner as the app gives them
        result = invoke(record["input"], history=memory.history_payload("", record["history"]))
    else:
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
        "models_used": result.get("models_used", []),   # which model answered each call (a fallback changes results)
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
        from shared.deepeval_gemini_model import GeminiDeepEvalModel

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


def _allow_ragas_import() -> None:
    """RAGAS 0.4.3 (the latest release) imports ChatVertexAI and VertexAI from langchain_community, which
    langchain-community 0.4 no longer ships, so `import ragas` raised ModuleNotFoundError. RAGAS only lists
    those two names among the classes it special-cases; it never builds one here. Inert stand-ins let it
    import. Harmless once RAGAS is fixed upstream (a real module/attribute is left alone)."""
    import sys
    import types

    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ImportError:
        stub = types.ModuleType("langchain_community.chat_models.vertexai")
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = stub

    import langchain_community.llms as community_llms

    if "VertexAI" not in vars(community_llms):
        try:
            from langchain_community.llms import VertexAI  # noqa: F401
        except ImportError:
            community_llms.VertexAI = type("VertexAI", (), {})


def _ragas_llm_for(spec: dict):
    """A native RAGAS LLM for one evaluation-tier model. RAGAS needs a
    provider client (not a text-in/text-out function), so it cannot go through
    llm_connection.generate(); this builds the client for the tier's model."""
    import os

    _allow_ragas_import()
    from ragas.llms import llm_factory

    provider, model = spec["provider"], spec["model"]
    api_key = os.environ[spec["config"]["api_key_env"]]

    # RAGAS scores through its ASYNC path (ascore -> agenerate), which refuses a synchronous client
    # ("Cannot use agenerate() with a synchronous client"), so every client here is an async one.
    if provider == "gemini":
        import openai

        # RAGAS's native Gemini wrapper is a synchronous client (and its source recommends this route):
        # Google's OpenAI-compatible endpoint, driven by an async OpenAI client, works with its async path.
        client = openai.AsyncOpenAI(
            api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return llm_factory(model, provider="openai", client=client)
    if provider == "openai":
        import openai

        return llm_factory(model, provider="openai", client=openai.AsyncOpenAI(api_key=api_key))
    if provider == "anthropic":
        import anthropic

        return llm_factory(model, provider="anthropic", client=anthropic.AsyncAnthropic(api_key=api_key))
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
    _allow_ragas_import()
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
    parser.add_argument("--golden-extra", default=None, metavar="PATH",
                        help="more golden questions to run after the main set (for example data/golden_set_m3.jsonl: tables, "
                             "images, a listing question, follow-ups)")
    parser.add_argument("--only-extra", action="store_true",
                        help="run ONLY the --golden-extra questions (a small trial: with --limit 3, three questions); "
                             "the main golden set is skipped")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()

    from retrieval.retrieval_graph import build_graph
    from retrieval.run_query import invoke_graph

    graph = build_graph()
    if args.only_extra:
        if not args.golden_extra:
            parser.error("--only-extra needs --golden-extra PATH")
        records = load_golden_set(args.golden_extra, limit=args.limit)   # a small trial: --limit applies to the extra file
    else:
        records = load_golden_set(limit=args.limit)
        if args.golden_extra:
            records += load_golden_set(args.golden_extra)
    usage_before = usage_tracker.snapshot()
    print(f"Running {len(records)} golden example(s) through the pipeline...")
    rows = run_pipeline(lambda q, history=None: invoke_graph(graph, q, history=history or "", use_cache=False), records)

    # The pipeline run above is the slow, quota-spending part. If a judge cannot score (every model in the
    # evaluation tier failed), keep the deterministic metrics and still log the run, instead of losing it all.
    judge_errors = {}
    for judge, enabled, score in (
        ("deepeval", args.llm_metrics in ("deepeval", "both"), score_deepeval),
        ("ragas", args.llm_metrics in ("ragas", "both"), score_ragas),
    ):
        if not enabled:
            continue
        print(f"Scoring with {judge}...")
        try:
            score(rows)
        except Exception as e:
            judge_errors[judge] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"   ({judge} scoring FAILED, continuing without its metrics: {judge_errors[judge]})")

    summary = aggregate(rows)
    for name, value in usage_tracker.delta(usage_before, usage_tracker.snapshot()).items():
        summary[f"usage_{name}"] = value
    print("\nAggregate metrics:")
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")

    if not args.no_mlflow:
        params = experiment_tracking.collect_versions(GOLDEN_SET_PATH)
        params["n_examples"] = len(rows)
        if args.golden_extra:
            params["golden_extra_md5"] = experiment_tracking.golden_set_version(args.golden_extra)
        params["llm_metrics"] = args.llm_metrics
        for judge, error in judge_errors.items():
            params[f"{judge}_error"] = error
        run_id = experiment_tracking.log_run("golden-set-eval", params, summary, rows)
        print(f"\nLogged MLflow run {run_id} (view: mlflow ui --backend-store-uri {experiment_tracking.MLRUNS_DIR.as_uri()})")


if __name__ == "__main__":
    main()
