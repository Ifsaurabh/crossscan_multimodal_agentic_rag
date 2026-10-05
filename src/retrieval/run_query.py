import sys

from shared import langfuse_client
from shared import llm_connection
from shared import query_guardrail
from retrieval import tracing
from retrieval.retrieval_graph import build_graph


def initial_state(query: str, history: str = "", notes: str = "", use_cache: bool = True) -> dict:
    # Redact BEFORE the query enters the graph: tracing records the graph's
    # input, and traces must never contain raw PII. The graph's input
    # guardrail node redacts again, which is harmless (idempotent).
    redacted_query, _ = query_guardrail.redact_pii(query)
    return {
        "history": history,
        "notes": notes,
        "use_cache": use_cache,
        "raw_query": redacted_query,
        "cleaned_query": None,
        "blocked": False,
        "block_reason": None,
        "cache_hit": False,
        "sub_queries": [],
        "final_answer": None,
        "guardrail_flags": [],
        "injection_score": None,
    }


def invoke_graph(graph, query: str, history: str = "", notes: str = "", use_cache: bool = True):
    """Run one query through the compiled graph, traced to Langfuse when it's
    enabled (no-op callbacks otherwise). `history`/`notes` carry chat context;
    `use_cache=False` (evaluation) skips the answer cache entirely."""
    tracing.reset()  # the seconds of this question, by step (db_s, embed_s, rerank_s, retrieval_s)
    llm_connection.start_recording_models()  # which model answered each call (a fallback changes the result)
    result = graph.invoke(
        initial_state(query, history, notes, use_cache),
        config={"callbacks": langfuse_client.get_callbacks()},
    )
    result["timings"] = tracing.snapshot()
    result["models_used"] = llm_connection.recorded_models()
    langfuse_client.flush()
    return result


def run_query(query: str):
    graph = build_graph()
    result = invoke_graph(graph, query)

    print(f"\nQuery: {query}")
    if result["blocked"]:
        print(f"BLOCKED: {result['block_reason']}")
        return result

    if result["cache_hit"]:
        print("(cache hit)")

    print(f"\nAnswer:\n{result['final_answer']}")
    if result.get("guardrail_flags"):
        print(f"\nGuardrail flags: {result['guardrail_flags']}")

    return result


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What accuracy did the lung cancer CNN model achieve?"
    run_query(query)
