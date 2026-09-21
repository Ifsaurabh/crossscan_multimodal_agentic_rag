import deepeval_gemini_model as dgm


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.last_call_kwargs = None

    def generate_content(self, **kwargs):
        self.last_call_kwargs = kwargs
        return FakeResponse(self.response_text)


class FakeClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)


def make_model(response_text="some output", monkeypatch=None, client=None):
    model = dgm.GeminiDeepEvalModel.__new__(dgm.GeminiDeepEvalModel)
    model.model_name = "gemini-3.6-flash"
    model.model = client or FakeClient(response_text)
    return model


def test_generate_returns_plain_text():
    model = make_model(response_text='{"sufficient": true}')

    result = model.generate("some prompt")

    assert result == '{"sufficient": true}'


def test_generate_does_not_accept_a_schema_kwarg():
    """DeepEval relies on this raising TypeError for custom models, so it
    can fall back to text-generation + its own JSON parsing instead of
    native structured-output support."""
    model = make_model()

    try:
        model.generate("prompt", schema=object())
        assert False, "expected TypeError"
    except TypeError:
        pass


async def _run_a_generate(model, prompt):
    return await model.a_generate(prompt)


def test_a_generate_delegates_to_generate():
    import asyncio

    model = make_model(response_text="async output")
    result = asyncio.run(_run_a_generate(model, "prompt"))

    assert result == "async output"


def test_generate_runs_on_the_evaluation_tier(monkeypatch):
    seen = {}

    class R:
        text = "judged"

    def fake_generate(system, user, **kwargs):
        seen.update(system=system, user=user, **kwargs)
        return R()

    monkeypatch.setattr(dgm.llm_connection, "generate", fake_generate)

    assert make_model().generate("score this") == "judged"
    assert seen["task"] == "deepeval_judge" and seen["system"] == "" and seen["user"] == "score this"


def test_get_model_name_returns_configured_name():
    model = make_model()
    assert model.get_model_name() == "gemini-3.6-flash"
