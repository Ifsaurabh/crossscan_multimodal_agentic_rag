import agent_transform_route as atr
import gemini_retry


def setup_function():
    gemini_retry._cache_registry.clear()


def test_the_planner_is_told_not_to_split_questions_that_retrieve_the_same_passages():
    text = atr.SYSTEM_INSTRUCTION

    assert "Do NOT split a question whose parts would retrieve the same passages" in text
    assert "ONE sub-question" in text


def test_the_planner_treats_a_single_papers_model_comparison_as_simple():
    text = atr.SYSTEM_INSTRUCTION

    assert "which of a paper's own models scored best" in text
    assert "comparison ACROSS different papers" in text


def test_build_prompt_includes_query():
    prompt = atr.build_prompt("What accuracy did the CNN model achieve?")
    assert "What accuracy did the CNN model achieve?" in prompt
    assert "RETRY" not in prompt


def test_build_prompt_includes_feedback_on_retry():
    feedback = {"missing": "accuracy numbers", "look_for": "results section"}
    prompt = atr.build_prompt("query", feedback=feedback)
    assert "RETRY" in prompt
    assert "accuracy numbers" in prompt
    assert "results section" in prompt


def test_parse_response_handles_plain_json():
    raw = '{"sub_queries": [{"sub_query": "q1", "variants": ["q1"], "needs_retrieval": true, "data_source": "vector", "search_mode": "simple", "images_required": false}]}'
    result = atr.parse_response(raw)
    assert result["sub_queries"][0]["sub_query"] == "q1"


def test_parse_response_strips_markdown_fence():
    raw = '```json\n{"sub_queries": []}\n```'
    result = atr.parse_response(raw)
    assert result == {"sub_queries": []}


def test_parse_response_returns_none_on_garbage():
    assert atr.parse_response("not json") is None


def test_user_content_has_no_history_or_notes_sections_by_default():
    content = atr.build_user_content("what is a CNN?")

    assert content == "User query: what is a CNN?"


def test_user_content_includes_history_and_notes_before_the_query():
    content = atr.build_user_content(
        "and its recall?", history="User: which YOLO?\nAssistant: YOLOv8", notes="- studies lung CT",
    )

    assert "Conversation so far:\nUser: which YOLO?" in content
    assert "User notes" in content and "- studies lung CT" in content
    assert content.index("Conversation so far") < content.index("User query: and its recall?")
    assert content.index("User notes") < content.index("Conversation so far")


def test_system_instruction_asks_for_a_complexity_label_in_the_json_format():
    assert "complexity" in atr.SYSTEM_INSTRUCTION
    assert '"complexity": "simple"' in atr.SYSTEM_INSTRUCTION
    assert 'When genuinely unsure, choose "complex"' in atr.SYSTEM_INSTRUCTION  # unsure still defaults to the stronger tier


def test_transform_and_route_runs_on_the_query_planner_task(monkeypatch):
    seen = {}

    class R:
        text = '{"sub_queries": []}'

    def fake_generate(system, user, **kwargs):
        seen.update(kwargs)
        return R()

    monkeypatch.setattr(atr.llm_connection, "generate", fake_generate)

    atr.transform_and_route("what is a CNN?")

    assert seen["task"] == "query_planner"


def test_system_instruction_tells_the_model_to_resolve_follow_ups():
    assert "self-contained" in atr.SYSTEM_INSTRUCTION
    assert "never as facts about the papers" in atr.SYSTEM_INSTRUCTION


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.last_prompt = None

    def generate_content(self, **kwargs):
        self.last_prompt = kwargs.get("contents")
        return FakeResponse(self.response_text)


class FakeCaches:
    """Simulates caching being unavailable (e.g. no billing enabled) -
    the common free-tier case - so transform_and_route falls back to
    inlining the instructions, same as it would in production."""
    def create(self, model, config):
        raise Exception("caching unavailable in test")


class FakeClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)
        self.caches = FakeCaches()


def test_transform_and_route_calls_gemini_and_parses():
    response_json = (
        '{"sub_queries": [{"sub_query": "which papers use CNN", "variants": ["which papers use CNN"], '
        '"needs_retrieval": true, "data_source": "graph", "search_mode": "simple", "images_required": false}]}'
    )
    fake_client = FakeClient(response_json)

    result = atr.transform_and_route("which papers use CNN", client=fake_client)

    assert result["sub_queries"][0]["data_source"] == "graph"
    assert "which papers use CNN" in fake_client.models.last_prompt
