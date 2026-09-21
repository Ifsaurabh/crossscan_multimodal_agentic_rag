import run_query


def test_initial_state_redacts_pii_before_the_graph_sees_it():
    state = run_query.initial_state("Contact me at jane.doe@example.com about CNNs")

    assert "jane.doe@example.com" not in state["raw_query"]
    assert "[REDACTED_EMAIL]" in state["raw_query"]


def test_initial_state_leaves_clean_queries_unchanged():
    state = run_query.initial_state("What accuracy did the CNN achieve?")

    assert state["raw_query"] == "What accuracy did the CNN achieve?"
    assert state["blocked"] is False
    assert state["sub_queries"] == []


def test_initial_state_carries_history_and_notes_defaulting_to_empty():
    plain = run_query.initial_state("q")
    chat = run_query.initial_state("q", history="User: hi", notes="- likes YOLO")

    assert plain["history"] == "" and plain["notes"] == ""
    assert chat["history"] == "User: hi" and chat["notes"] == "- likes YOLO"


def test_use_cache_defaults_on_and_can_be_switched_off_for_evaluation(monkeypatch):
    seen = []

    class FakeGraph:
        def invoke(self, state, config=None):
            seen.append(state["use_cache"])
            return {}

    monkeypatch.setattr(run_query.langfuse_client, "get_callbacks", lambda: [])
    monkeypatch.setattr(run_query.langfuse_client, "flush", lambda: None)

    run_query.invoke_graph(FakeGraph(), "q")
    run_query.invoke_graph(FakeGraph(), "q", use_cache=False)

    assert seen == [True, False]


def test_invoke_graph_forwards_history_and_notes(monkeypatch):
    seen = {}

    class FakeGraph:
        def invoke(self, state, config=None):
            seen["state"] = state
            return {}

    monkeypatch.setattr(run_query.langfuse_client, "get_callbacks", lambda: [])
    monkeypatch.setattr(run_query.langfuse_client, "flush", lambda: None)

    run_query.invoke_graph(FakeGraph(), "and its recall?", history="User: which YOLO?", notes="- studies CT")

    assert seen["state"]["history"] == "User: which YOLO?"
    assert seen["state"]["notes"] == "- studies CT"


def test_invoke_graph_passes_callbacks_and_redacted_input(monkeypatch):
    seen = {}

    class FakeGraph:
        def invoke(self, state, config=None):
            seen["state"] = state
            seen["config"] = config
            return {"final_answer": "ok"}

    monkeypatch.setattr(run_query.langfuse_client, "get_callbacks", lambda: ["CB"])
    monkeypatch.setattr(run_query.langfuse_client, "flush", lambda: None)

    result = run_query.invoke_graph(FakeGraph(), "mail bob@example.org")

    assert result == {"final_answer": "ok"}
    assert seen["config"] == {"callbacks": ["CB"]}
    assert "bob@example.org" not in seen["state"]["raw_query"]
