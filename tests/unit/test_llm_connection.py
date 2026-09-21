import sys
import types

import pytest

import llm_connection as lc
import usage_tracker


class HttpError(Exception):
    def __init__(self, code=None, message="boom", status_code=None):
        super().__init__(message)
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


def setup_function():
    lc.reset_cooldowns()
    lc._clients.clear()
    usage_tracker.reset()


@pytest.fixture
def keys(monkeypatch):
    for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "test-key")


class Recorder:
    """Fake adapters. `behaviours` maps a MODEL name (or "*" for any other
    model) to an exception to raise or the text to reply with. Records every
    call as (provider, model, system, user, client)."""

    def __init__(self, monkeypatch, behaviours, providers=("gemini", "anthropic", "openai")):
        self.calls = []
        self.behaviours = behaviours
        for provider in providers:
            monkeypatch.setitem(lc._ADAPTERS, provider, self._make(provider))

    def _make(self, provider):
        def adapter(config, model, system, user, client=None):
            self.calls.append({
                "provider": provider, "model": model, "system": system, "user": user, "client": client,
                "config": config,
            })
            behaviour = self.behaviours.get(model, self.behaviours.get("*", f"reply from {model}"))
            if isinstance(behaviour, Exception):
                raise behaviour
            return lc.LLMResult(text=behaviour, provider=provider, model=model)
        return adapter

    def models_called(self):
        return [c["model"] for c in self.calls]


FAST_PRIMARY, FAST_FALLBACK = [s["model"] for s in lc.TIERS["fast"]]
REASONING_PRIMARY, REASONING_FALLBACK = [s["model"] for s in lc.TIERS["reasoning"]]
EVAL_PRIMARY, EVAL_FALLBACK = [s["model"] for s in lc.TIERS["evaluation"]]


# ---------- the agreed settings ----------

def test_default_tiers_are_the_agreed_google_only_flash_models():
    assert lc.TIERS == {
        "fast": [
            {"provider": "gemini", "model": "gemini-3.6-flash"},
            {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
        ],
        "reasoning": [
            {"provider": "gemini", "model": "gemini-3.8-flash"},
            {"provider": "gemini", "model": "gemini-3.7-flash"},
        ],
        "evaluation": [
            {"provider": "gemini", "model": "gemini-3.7-flash"},
            {"provider": "gemini", "model": "gemini-3.5-flash"},
        ],
    }


def test_every_tier_has_a_primary_and_a_fallback_and_only_google_is_used():
    for tier, models in lc.TIERS.items():
        assert len(models) >= lc.MIN_MODELS_PER_TIER, tier
        assert {m["provider"] for m in models} == {"gemini"}


def test_default_task_to_tier_table():
    assert lc.TASK_TIERS == {
        "query_planner": "fast",
        "quality_check": "fast",
        "general_knowledge": "fast",
        "conversation_summary": "fast",
        "answer_simple": "fast",
        "answer_complex": "reasoning",
        "deepeval_judge": "evaluation",
        "ragas_judge": "evaluation",
        "online_judge": "evaluation",
    }


def test_api_keys_are_never_stored_in_the_settings():
    for config in lc.PROVIDERS.values():
        assert set(config) <= {"api_key_env", "timeout_seconds", "max_output_tokens"}
        assert config["api_key_env"].endswith("_API_KEY")


def test_default_settings_validate():
    lc.validate_settings()  # must not raise


def test_validate_rejects_a_tier_without_a_fallback(monkeypatch):
    monkeypatch.setattr(lc, "TIERS", {**lc.TIERS, "fast": [lc.TIERS["fast"][0]]})
    with pytest.raises(ValueError, match="at least 2"):
        lc.validate_settings()


def test_validate_rejects_the_same_model_twice_in_a_tier(monkeypatch):
    twice = {"provider": "gemini", "model": "gemini-3.6-flash"}
    monkeypatch.setattr(lc, "TIERS", {**lc.TIERS, "fast": [twice, dict(twice)]})
    with pytest.raises(ValueError, match="twice"):
        lc.validate_settings()


def test_validate_rejects_unknown_provider_and_unknown_task_tier(monkeypatch):
    monkeypatch.setattr(lc, "TIERS", {**lc.TIERS, "fast": [
        {"provider": "nonexistent", "model": "a"}, {"provider": "gemini", "model": "b"},
    ]})
    with pytest.raises(ValueError, match="unknown provider"):
        lc.validate_settings()

    monkeypatch.undo()
    monkeypatch.setattr(lc, "TASK_TIERS", {"query_planner": "no-such-tier"})
    with pytest.raises(ValueError, match="unknown tier"):
        lc.validate_settings()


# ---------- tier / task resolution ----------

def test_resolve_tier_from_tier_or_task():
    assert lc.resolve_tier(tier="reasoning") == "reasoning"
    assert lc.resolve_tier(task="answer_complex") == "reasoning"
    assert lc.resolve_tier(task="query_planner") == "fast"
    assert lc.resolve_tier(tier="fast", task="answer_complex") == "fast"  # explicit tier wins


def test_resolve_tier_rejects_unknown_and_missing():
    with pytest.raises(ValueError, match="Unknown tier"):
        lc.resolve_tier(tier="turbo")
    with pytest.raises(ValueError, match="Unknown task"):
        lc.resolve_tier(task="make_coffee")
    with pytest.raises(ValueError, match="needs a tier"):
        lc.resolve_tier()


# ---------- fallback within a tier ----------

def test_primary_of_the_requested_tier_answers(monkeypatch, keys):
    rec = Recorder(monkeypatch, {})

    result = lc.generate("sys", "hello", tier="reasoning")

    assert result.model == REASONING_PRIMARY and result.tier == "reasoning"
    assert result.text == f"reply from {REASONING_PRIMARY}"
    assert result.fallback_used is False and result.attempts == []
    assert rec.models_called() == [REASONING_PRIMARY]
    assert rec.calls[0]["system"] == "sys" and rec.calls[0]["user"] == "hello"


def test_a_task_uses_its_tiers_models(monkeypatch, keys):
    rec = Recorder(monkeypatch, {})

    lc.generate("s", "q", task="answer_complex")
    lc.generate("s", "q", task="quality_check")
    lc.generate("s", "q", task="deepeval_judge")

    assert rec.models_called() == [REASONING_PRIMARY, FAST_PRIMARY, EVAL_PRIMARY]


def test_falls_back_to_the_next_model_in_the_same_tier(monkeypatch, keys):
    rec = Recorder(monkeypatch, {FAST_PRIMARY: HttpError(429, "RESOURCE_EXHAUSTED")})

    result = lc.generate("s", "q", tier="fast")

    assert result.model == FAST_FALLBACK and result.tier == "fast"
    assert result.fallback_used is True
    assert result.attempts[0]["model"] == FAST_PRIMARY
    assert result.attempts[0]["status"] == "failed" and result.attempts[0]["kind"] == "quota"
    assert rec.models_called() == [FAST_PRIMARY, FAST_FALLBACK]


def test_a_failed_model_is_skipped_during_its_cooldown(monkeypatch, keys):
    rec = Recorder(monkeypatch, {FAST_PRIMARY: HttpError(429)})

    lc.generate("s", "first", tier="fast")
    second = lc.generate("s", "second", tier="fast")

    assert rec.models_called() == [FAST_PRIMARY, FAST_FALLBACK, FAST_FALLBACK]  # primary not retried
    assert second.attempts[0]["status"] == "skipped" and "cooling down" in second.attempts[0]["detail"]


def test_cooldown_is_per_model_and_shared_by_every_tier_that_lists_that_model(monkeypatch, keys):
    # gemini-3.7-flash is reasoning's fallback AND evaluation's primary.
    assert REASONING_FALLBACK == EVAL_PRIMARY
    rec = Recorder(monkeypatch, {REASONING_PRIMARY: HttpError(503), REASONING_FALLBACK: HttpError(429)})

    with pytest.raises(lc.AllProvidersFailed):
        lc.generate("s", "q", tier="reasoning")
    result = lc.generate("s", "q", tier="evaluation")

    assert result.model == EVAL_FALLBACK  # the shared model is still on cooldown
    assert result.attempts[0]["model"] == EVAL_PRIMARY and result.attempts[0]["status"] == "skipped"
    assert rec.models_called().count(EVAL_PRIMARY) == 1  # only the earlier reasoning attempt


def test_a_tier_never_spills_into_another_tier(monkeypatch, keys):
    rec = Recorder(monkeypatch, {FAST_PRIMARY: HttpError(429), FAST_FALLBACK: HttpError(429)})

    with pytest.raises(lc.AllProvidersFailed) as excinfo:
        lc.generate("s", "q", tier="fast")

    assert excinfo.value.tier == "fast"
    assert [a["model"] for a in excinfo.value.attempts] == [FAST_PRIMARY, FAST_FALLBACK]
    assert set(rec.models_called()) == {FAST_PRIMARY, FAST_FALLBACK}  # no reasoning/evaluation model touched


def test_all_failed_error_lists_every_attempt_and_chains_the_last_error(monkeypatch, keys):
    last = HttpError(500, "fallback down")
    Recorder(monkeypatch, {REASONING_PRIMARY: HttpError(429), REASONING_FALLBACK: last})

    with pytest.raises(lc.AllProvidersFailed) as excinfo:
        lc.generate("s", "q", tier="reasoning")

    message = str(excinfo.value)
    assert "reasoning" in message and REASONING_PRIMARY in message and REASONING_FALLBACK in message
    assert excinfo.value.__cause__ is last


def test_without_an_api_key_every_model_is_skipped(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    rec = Recorder(monkeypatch, {})

    with pytest.raises(lc.AllProvidersFailed) as excinfo:
        lc.generate("s", "q", tier="fast")

    assert rec.calls == []
    assert all(a["status"] == "skipped" and "GEMINI_API_KEY" in a["detail"] for a in excinfo.value.attempts)


def test_gemini_client_override_bypasses_the_key_check_and_is_passed_through(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    sentinel = object()
    rec = Recorder(monkeypatch, {})

    result = lc.generate("s", "q", tier="fast", client=sentinel)

    assert result.model == FAST_PRIMARY
    assert rec.calls[0]["client"] is sentinel


def test_a_model_with_a_healthy_fallback_gets_the_short_retry_budget(monkeypatch, keys):
    rec = Recorder(monkeypatch, {})

    lc.generate("s", "q", tier="fast")

    assert rec.calls[0]["config"]["max_retries"] == lc.FAILOVER_RETRIES == 2  # one wait, then fail over


def test_the_last_usable_model_keeps_the_full_patience(monkeypatch, keys):
    rec = Recorder(monkeypatch, {FAST_PRIMARY: lc.EmptyResponse("no text")})

    lc.generate("s", "q", tier="fast")

    assert [c["config"]["max_retries"] for c in rec.calls] == [lc.FAILOVER_RETRIES, lc.FULL_RETRIES]
    assert lc.FULL_RETRIES == 4


def test_when_the_fallback_is_cooling_down_the_primary_keeps_the_full_patience(monkeypatch, keys):
    rec = Recorder(monkeypatch, {})
    lc._start_cooldown("gemini", FAST_FALLBACK, "quota")  # nobody to fail over to

    lc.generate("s", "q", tier="fast")

    assert rec.calls[0]["model"] == FAST_PRIMARY
    assert rec.calls[0]["config"]["max_retries"] == lc.FULL_RETRIES


def test_the_retry_budget_never_changes_the_shared_provider_settings(monkeypatch, keys):
    Recorder(monkeypatch, {})

    lc.generate("s", "q", tier="fast")

    assert "max_retries" not in lc.PROVIDERS["gemini"]


def test_the_gemini_adapter_passes_its_retry_budget_on(monkeypatch):
    import gemini_retry

    seen = {}
    monkeypatch.setattr(
        gemini_retry, "call_with_retry",
        lambda client, model, contents, max_retries=None: seen.update(retries=max_retries) or FakeGeminiResponse("ok"),
    )

    lc._call_gemini({**CONFIG["gemini"], "max_retries": 2}, "gemini-3.7-flash", "", "p", client=object())

    assert seen["retries"] == 2


def test_missing_sdk_skips_without_cooldown_and_continues(monkeypatch, keys):
    calls = []

    def adapter(config, model, system, user, client=None):
        calls.append(model)
        if model == FAST_PRIMARY:
            raise lc.ProviderUnavailable("sdk not installed")
        return lc.LLMResult(text="ok", provider="gemini", model=model)

    monkeypatch.setitem(lc._ADAPTERS, "gemini", adapter)

    result = lc.generate("s", "q", tier="fast")

    assert result.model == FAST_FALLBACK
    assert result.attempts[0]["status"] == "skipped"
    assert lc._cooldown_remaining("gemini", FAST_PRIMARY) == 0


def test_empty_response_falls_back_but_does_not_cool_the_model_down(monkeypatch, keys):
    Recorder(monkeypatch, {FAST_PRIMARY: lc.EmptyResponse("no text")})

    result = lc.generate("s", "q", tier="fast")

    assert result.model == FAST_FALLBACK
    assert result.attempts[0]["kind"] == "other"
    assert lc._cooldown_remaining("gemini", FAST_PRIMARY) == 0


def test_a_tier_can_later_be_pointed_at_another_provider(monkeypatch, keys):
    """The point of the design: switching vendors is a settings edit."""
    monkeypatch.setattr(lc, "TIERS", {**lc.TIERS, "reasoning": [
        {"provider": "gemini", "model": "gemini-x"},
        {"provider": "openai", "model": "some-openai-model"},
    ]})
    rec = Recorder(monkeypatch, {"gemini-x": HttpError(429)})

    result = lc.generate("s", "q", tier="reasoning")

    assert result.provider == "openai" and result.model == "some-openai-model"
    assert [c["provider"] for c in rec.calls] == ["gemini", "openai"]
    assert rec.calls[1]["client"] is None  # the Gemini client override is only for Gemini


# ---------- native-client callers (RAGAS) ----------

def test_ready_models_lists_the_tier_in_order_with_their_config(keys):
    ready = lc.ready_models("evaluation")

    assert [(m["provider"], m["model"]) for m in ready] == [("gemini", EVAL_PRIMARY), ("gemini", EVAL_FALLBACK)]
    assert ready[0]["config"]["api_key_env"] == "GEMINI_API_KEY"


def test_ready_models_excludes_cooled_down_and_keyless_models(monkeypatch, keys):
    lc._start_cooldown("gemini", EVAL_PRIMARY, "quota")
    assert [m["model"] for m in lc.ready_models("evaluation")] == [EVAL_FALLBACK]

    monkeypatch.delenv("GEMINI_API_KEY")
    assert lc.ready_models("evaluation") == []


def test_record_failure_starts_the_same_cooldown_generate_would(keys):
    kind = lc.record_failure("gemini", EVAL_PRIMARY, HttpError(429, "RESOURCE_EXHAUSTED"))

    assert kind == "quota"
    assert lc._cooldown_remaining("gemini", EVAL_PRIMARY) > 0
    assert [m["model"] for m in lc.ready_models("evaluation")] == [EVAL_FALLBACK]


# ---------- error classification and cooldowns ----------

@pytest.mark.parametrize("error,expected", [
    (HttpError(429), "quota"),
    (HttpError(message="RESOURCE_EXHAUSTED: quota exceeded"), "quota"),
    (HttpError(status_code=429), "quota"),
    (HttpError(message="Rate limit reached for gpt-4o-mini"), "quota"),
    (HttpError(401), "auth"),
    (HttpError(status_code=403), "auth"),
    (HttpError(message="invalid x-api-key"), "auth"),
    (HttpError(503), "unavailable"),
    (HttpError(status_code=529), "unavailable"),
    (TimeoutError("request timed out"), "unavailable"),
    (ConnectionError("connection reset"), "unavailable"),
    (ValueError("something else"), "other"),
    (HttpError(400, "bad request"), "other"),
])
def test_classify_error(error, expected):
    assert lc.classify_error(error) == expected


def test_cooldown_lengths_follow_the_error_kind_and_are_per_model():
    lc._start_cooldown("gemini", "m-quota", "quota", now=1000.0)
    lc._start_cooldown("gemini", "m-down", "unavailable", now=1000.0)
    lc._start_cooldown("gemini", "m-other", "other", now=1000.0)

    assert lc._cooldown_remaining("gemini", "m-quota", now=1000.0) == lc.COOLDOWN_SECONDS["quota"]
    assert lc._cooldown_remaining("gemini", "m-down", now=1000.0) == lc.COOLDOWN_SECONDS["unavailable"]
    assert lc._cooldown_remaining("gemini", "m-other", now=1000.0) == 0
    assert lc._cooldown_remaining("gemini", "untouched", now=1000.0) == 0
    assert lc._cooldown_remaining("gemini", "m-quota", now=1000.0 + lc.COOLDOWN_SECONDS["quota"] + 1) == 0


def test_reset_cooldowns_clears_everything():
    lc._start_cooldown("gemini", "m", "quota")
    lc.reset_cooldowns()
    assert lc._cooldown_remaining("gemini", "m") == 0


# ---------- status and snapshot ----------

def test_status_reports_every_tier_and_model(monkeypatch, keys):
    lc._start_cooldown("gemini", FAST_PRIMARY, "quota")

    report = lc.status()

    assert list(report) == ["fast", "reasoning", "evaluation"]
    fast = report["fast"]
    assert [e["model"] for e in fast] == [FAST_PRIMARY, FAST_FALLBACK]
    assert fast[0]["ready"] is False and "cooling down" in fast[0]["reason"]
    assert fast[1]["ready"] is True and fast[1]["reason"] is None


def test_status_flags_missing_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert all(not e["ready"] and "GEMINI_API_KEY" in e["reason"] for tier in lc.status().values() for e in tier)


def test_config_snapshot_lists_each_tiers_models_in_order():
    snapshot = lc.config_snapshot()
    assert snapshot["llm_tier_fast"] == "gemini:gemini-3.6-flash,gemini:gemini-3.5-flash-lite"
    assert snapshot["llm_tier_reasoning"] == "gemini:gemini-3.8-flash,gemini:gemini-3.7-flash"
    assert snapshot["llm_tier_evaluation"] == "gemini:gemini-3.7-flash,gemini:gemini-3.5-flash"


# ---------- real adapters against fake SDKs / clients ----------

class FakeGeminiResponse:
    def __init__(self, text, prompt=10, output=5):
        self.text = text
        self.usage_metadata = types.SimpleNamespace(
            prompt_token_count=prompt, candidates_token_count=output, cached_content_token_count=0,
        )


class FakeGeminiClient:
    def __init__(self, text):
        self.text = text
        self.calls = []
        self.models = self
        self.caches = types.SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(Exception("no caching in test")))

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeGeminiResponse(self.text)


CONFIG = lc.PROVIDERS


def test_gemini_adapter_uses_the_tier_model_and_inlines_the_system_instruction():
    import gemini_retry

    gemini_retry._cache_registry.clear()
    client = FakeGeminiClient("  answer  ")

    result = lc._call_gemini(CONFIG["gemini"], "gemini-3.8-flash", "the system", "the user", client=client)

    assert result.text == "answer" and result.provider == "gemini" and result.model == "gemini-3.8-flash"
    assert result.prompt_tokens == 10 and result.output_tokens == 5
    assert client.calls[0]["model"] == "gemini-3.8-flash"
    assert "the system" in client.calls[0]["contents"] and "the user" in client.calls[0]["contents"]


def test_gemini_adapter_without_system_instruction_sends_the_prompt_only():
    client = FakeGeminiClient("ok")

    lc._call_gemini(CONFIG["gemini"], "gemini-3.7-flash", "", "just the prompt", client=client)

    assert client.calls[0]["contents"] == "just the prompt"


def test_gemini_adapter_empty_text_raises_empty_response():
    with pytest.raises(lc.EmptyResponse):
        lc._call_gemini(CONFIG["gemini"], "gemini-3.7-flash", "", "p", client=FakeGeminiClient("   "))


def test_gemini_calls_are_not_double_counted_in_usage():
    """gemini_retry records Gemini usage itself; the adapter must not add it again."""
    import gemini_retry

    gemini_retry._cache_registry.clear()
    lc._call_gemini(CONFIG["gemini"], "gemini-3.7-flash", "", "p", client=FakeGeminiClient("ok"))

    snapshot = usage_tracker.snapshot()
    assert snapshot["calls"] == 1 and snapshot["prompt_tokens"] == 10


def install_fake_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_anthropic_adapter_uses_the_given_model_and_records_usage(monkeypatch):
    created = {}

    class FakeMessages:
        def create(self, **kwargs):
            created.update(kwargs)
            block = types.SimpleNamespace(type="text", text="claude says hi")
            usage = types.SimpleNamespace(input_tokens=120, output_tokens=30, cache_read_input_tokens=7)
            return types.SimpleNamespace(content=[block, types.SimpleNamespace(type="tool_use")], usage=usage)

    class FakeAnthropic:
        def __init__(self, api_key, timeout):
            created["api_key"], created["timeout"] = api_key, timeout
            self.messages = FakeMessages()

    install_fake_module(monkeypatch, "anthropic", Anthropic=FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    result = lc._call_anthropic(CONFIG["anthropic"], "some-claude-model", "be brief", "question?")

    assert result.text == "claude says hi" and result.model == "some-claude-model"
    assert created["model"] == "some-claude-model" and created["system"] == "be brief"
    assert created["messages"] == [{"role": "user", "content": "question?"}]
    assert created["max_tokens"] == CONFIG["anthropic"]["max_output_tokens"]
    snapshot = usage_tracker.snapshot()
    assert (snapshot["prompt_tokens"], snapshot["output_tokens"], snapshot["cached_tokens"]) == (120, 30, 7)


def test_anthropic_adapter_without_the_sdk_raises_provider_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` raise ImportError

    with pytest.raises(lc.ProviderUnavailable, match="pip install anthropic"):
        lc._call_anthropic(CONFIG["anthropic"], "m", "s", "q")


def test_openai_adapter_uses_the_given_model_and_records_usage(monkeypatch):
    created = {}

    class FakeCompletions:
        def create(self, **kwargs):
            created.update(kwargs)
            usage = types.SimpleNamespace(
                prompt_tokens=200, completion_tokens=40,
                prompt_tokens_details=types.SimpleNamespace(cached_tokens=11),
            )
            message = types.SimpleNamespace(content=" gpt says hi ")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage)

    class FakeOpenAI:
        def __init__(self, api_key, timeout):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    install_fake_module(monkeypatch, "openai", OpenAI=FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    result = lc._call_openai(CONFIG["openai"], "some-openai-model", "be brief", "question?")

    assert result.text == "gpt says hi" and result.model == "some-openai-model"
    assert created["model"] == "some-openai-model"
    assert created["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "question?"}]
    assert created["max_completion_tokens"] == CONFIG["openai"]["max_output_tokens"]
    snapshot = usage_tracker.snapshot()
    assert (snapshot["prompt_tokens"], snapshot["output_tokens"], snapshot["cached_tokens"]) == (200, 40, 11)


def test_openai_adapter_empty_content_raises_empty_response(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            message = types.SimpleNamespace(content=None)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

    install_fake_module(
        monkeypatch, "openai",
        OpenAI=lambda api_key, timeout: types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions())),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    with pytest.raises(lc.EmptyResponse):
        lc._call_openai(CONFIG["openai"], "m", "", "q")
