import agent_quality_generate as aqg
import gemini_retry


def setup_function():
    gemini_retry._cache_registry.clear()


def test_format_context_includes_source_and_page():
    chunks = [{"source_pdf": "a.pdf", "page_start": 4, "parent_text": "Full parent text here."}]
    formatted = aqg.format_context(chunks)
    assert "[a.pdf, p.4]" in formatted
    assert "Full parent text here." in formatted


def test_format_context_falls_back_to_text_field():
    chunks = [{"source_pdf": "a.pdf", "page_start": 1, "text": "child text"}]
    formatted = aqg.format_context(chunks)
    assert "child text" in formatted


def test_parse_json_response_handles_markdown_fence():
    raw = '```json\n{"sufficient": true, "missing": "", "look_for": ""}\n```'
    result = aqg.parse_json_response(raw)
    assert result["sufficient"] is True


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text

    def generate_content(self, **kwargs):
        return FakeResponse(self.response_text)


class FakeCaches:
    """Simulates caching being unavailable (e.g. no billing enabled) -
    the common free-tier case - so the agent falls back to inlining the
    instructions, same as it would in production."""
    def create(self, model, config):
        raise Exception("caching unavailable in test")


class FakeClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)
        self.caches = FakeCaches()


def test_check_quality_returns_insufficient_for_empty_chunks():
    result = aqg.check_quality("some query", [], client=FakeClient("unused"))
    assert result["sufficient"] is False
    assert "no results" in result["missing"]


def test_check_quality_parses_gemini_response():
    fake_client = FakeClient('{"sufficient": true, "missing": "", "look_for": ""}')
    chunks = [{"source_pdf": "a.pdf", "page_start": 1, "text": "some content"}]

    result = aqg.check_quality("query", chunks, client=fake_client)

    assert result["sufficient"] is True


def test_check_quality_handles_unparseable_response():
    fake_client = FakeClient("not valid json")
    chunks = [{"source_pdf": "a.pdf", "page_start": 1, "text": "some content"}]

    result = aqg.check_quality("query", chunks, client=fake_client)

    assert result["sufficient"] is False
    assert "could not judge" in result["missing"]


def test_generate_answer_calls_gemini():
    fake_client = FakeClient("98% accuracy [a.pdf, p.4]")
    chunks = [{"source_pdf": "a.pdf", "page_start": 4, "text": "The model achieved 98% accuracy"}]

    answer = aqg.generate_answer("what accuracy?", chunks, client=fake_client)

    assert answer == "98% accuracy [a.pdf, p.4]"


def test_generate_general_knowledge_answer_prepends_transparency_label():
    fake_client = FakeClient("CNN stands for Convolutional Neural Network.")

    answer = aqg.generate_general_knowledge_answer("what does CNN stand for?", client=fake_client)

    assert answer.startswith(aqg.NOT_FROM_KNOWLEDGE_BASE_LABEL)
    assert "Convolutional Neural Network" in answer


def test_generate_answer_handles_empty_chunks():
    fake_client = FakeClient("I don't have information on this.")
    answer = aqg.generate_answer("query", [], client=fake_client)
    assert answer == "I don't have information on this."


def test_answer_task_maps_complexity_to_a_task():
    assert aqg.answer_task("simple") == "answer_simple"
    assert aqg.answer_task("complex") == "answer_complex"
    assert aqg.answer_task(None) == "answer_complex"      # a missing label counts as complex
    assert aqg.answer_task("whatever") == "answer_complex"  # so does an unrecognised one


class FakeResult:
    text = "an answer"


def capture_tasks(monkeypatch):
    seen = []

    def fake_generate(system, user, **kwargs):
        seen.append(kwargs["task"])
        return FakeResult()

    monkeypatch.setattr(aqg.llm_connection, "generate", fake_generate)
    return seen


def test_generate_answer_uses_the_tier_for_its_complexity(monkeypatch):
    seen = capture_tasks(monkeypatch)

    aqg.generate_answer("q", [], complexity="simple")
    aqg.generate_answer("q", [], complexity="complex")
    aqg.generate_answer("q", [])

    assert seen == ["answer_simple", "answer_complex", "answer_complex"]


def test_other_agent_calls_use_their_own_tasks(monkeypatch):
    seen = capture_tasks(monkeypatch)
    chunks = [{"source_pdf": "a.pdf", "page_start": 1, "text": "content"}]

    aqg.check_quality("q", chunks)
    aqg.generate_general_knowledge_answer("what does CNN stand for?")

    assert seen == ["quality_check", "general_knowledge"]
