from google.genai.errors import ServerError

import gemini_retry


class FlakyModels:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0
        self.last_call_kwargs = None

    def generate_content(self, **kwargs):
        self.calls += 1
        self.last_call_kwargs = kwargs
        if self.calls <= self.fail_times:
            raise ServerError(503, {"error": {"message": "busy"}}, None)
        return "success"


class FlakyClient:
    def __init__(self, fail_times=0, caches=None):
        self.models = FlakyModels(fail_times)
        self.caches = caches


class FakeCaches:
    def __init__(self, should_fail=False, cache_name="cachedContents/abc123"):
        self.should_fail = should_fail
        self.cache_name = cache_name
        self.create_calls = 0

    def create(self, model, config):
        self.create_calls += 1
        if self.should_fail:
            raise Exception("billing not enabled")

        class FakeCache:
            name = self.cache_name
        return FakeCache()


def setup_function():
    gemini_retry._cache_registry.clear()


def test_call_with_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr(gemini_retry.time, "sleep", lambda s: None)
    client = FlakyClient(fail_times=2)

    result = gemini_retry.call_with_retry(client, "model", "prompt")

    assert result == "success"
    assert client.models.calls == 3


def test_call_with_retry_raises_after_max_attempts(monkeypatch):
    monkeypatch.setattr(gemini_retry.time, "sleep", lambda s: None)
    client = FlakyClient(fail_times=10)

    try:
        gemini_retry.call_with_retry(client, "model", "prompt", max_retries=3)
        assert False, "expected ServerError to be raised"
    except ServerError:
        pass

    assert client.models.calls == 3


# Big enough to be worth caching (Gemini's minimum is 1,024 tokens, estimated here from the text length).
BIG = "fixed instructions. " * 300


def test_call_with_cache_uses_cache_when_available(monkeypatch):
    client = FlakyClient(caches=FakeCaches(should_fail=False))

    result = gemini_retry.call_with_cache(client, "model", BIG, "the query")

    assert result == "success"
    assert client.caches.create_calls == 1
    # generate_content was called with cached_content config, not inlined instructions
    assert "config" in client.models.last_call_kwargs
    assert client.models.last_call_kwargs["contents"] == "the query"


def test_call_with_cache_falls_back_when_cache_creation_fails(monkeypatch):
    client = FlakyClient(caches=FakeCaches(should_fail=True))

    result = gemini_retry.call_with_cache(client, "model", BIG, "the query")

    assert result == "success"
    # fell back to inlining the instructions into contents, no config
    assert "config" not in client.models.last_call_kwargs or client.models.last_call_kwargs.get("config") is None
    assert BIG in client.models.last_call_kwargs["contents"]
    assert "the query" in client.models.last_call_kwargs["contents"]


def test_call_with_cache_only_attempts_cache_creation_once_per_instruction(monkeypatch):
    fake_caches = FakeCaches(should_fail=True)
    client = FlakyClient(caches=fake_caches)

    gemini_retry.call_with_cache(client, "model", BIG, "query one")
    gemini_retry.call_with_cache(client, "model", BIG, "query two")

    assert fake_caches.create_calls == 1  # not retried on the second call


def test_small_instructions_never_spend_a_request_on_a_doomed_cache_attempt(monkeypatch):
    fake_caches = FakeCaches(should_fail=False)
    client = FlakyClient(caches=fake_caches)

    result = gemini_retry.call_with_cache(client, "model", "a short instruction", "the query")

    assert result == "success"
    assert fake_caches.create_calls == 0  # Gemini would refuse < 1,024 tokens; we know that without asking
    assert "a short instruction" in client.models.last_call_kwargs["contents"]  # inlined, as before


def test_the_size_threshold_is_about_1024_tokens_of_text():
    assert gemini_retry.too_small_to_cache("x" * 4000) is True
    assert gemini_retry.too_small_to_cache("x" * 4096) is False


def test_call_with_cache_retries_on_server_error(monkeypatch):
    monkeypatch.setattr(gemini_retry.time, "sleep", lambda s: None)
    client = FlakyClient(fail_times=2, caches=FakeCaches(should_fail=True))

    result = gemini_retry.call_with_cache(client, "model", "instructions", "query")

    assert result == "success"
    assert client.models.calls == 3


def test_a_busy_model_message_names_the_model_and_the_real_error(monkeypatch, capsys):
    monkeypatch.setattr(gemini_retry.time, "sleep", lambda s: None)
    client = FlakyClient(fail_times=1)

    gemini_retry.call_with_retry(client, "gemini-3.8-flash", "prompt")

    out = capsys.readouterr().out
    assert "gemini-3.8-flash" in out and "503" in out and "retrying in 2s" in out


def test_a_smaller_retry_budget_gives_up_sooner_with_a_single_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(gemini_retry.time, "sleep", lambda s: waits.append(s))
    client = FlakyClient(fail_times=10)

    try:
        gemini_retry.call_with_cache(client, "model", "instructions", "query", max_retries=2)
        assert False, "expected ServerError"
    except ServerError:
        pass

    assert client.models.calls == 2 and waits == [2]  # one retry, one 2-second wait, then the caller can fail over


def test_the_default_is_still_four_attempts():
    assert gemini_retry.DEFAULT_MAX_RETRIES == 4
