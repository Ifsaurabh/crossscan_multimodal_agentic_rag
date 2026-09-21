"""Live smoke tests: the REAL services, nothing faked.

Everything in tests/live/ is skipped unless asked for:
    python -m pytest tests/live --run-live        (or RUN_LIVE=1)

These spend a little Gemini quota, and need Neo4j running and the graph loaded,
so they belong on a manual button or a weekly schedule - never on every push.
A failure here means a service, a key, a model name or the data is broken, not
that the code logic is wrong (the unit tests cover that).
"""
import pytest

pytestmark = pytest.mark.live


def test_the_fast_tier_answers_with_a_real_model():
    import llm_connection

    result = llm_connection.generate(
        "Reply with exactly one word.", "Say: ready", task="conversation_summary",
    )

    assert result.text.strip()
    assert result.tier == "fast"
    assert result.prompt_tokens > 0


def test_neo4j_is_reachable_and_the_graph_is_loaded():
    import graph_db

    driver = graph_db.get_driver()
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    finally:
        driver.close()

    assert nodes > 0, "Neo4j is up but empty: run setup_graph_db.py and load_graph_db.py"


def test_a_real_question_flows_through_the_whole_pipeline():
    import query_cache  # noqa: F401  (the graph's cache layer must import cleanly)
    from retrieval_graph import build_graph
    from run_query import invoke_graph

    result = invoke_graph(build_graph(), "What accuracy did the CNN model achieve?", use_cache=False)

    assert result["blocked"] is False
    assert result["final_answer"]
    assert result["sub_queries"], "the planner produced no sub-questions"
