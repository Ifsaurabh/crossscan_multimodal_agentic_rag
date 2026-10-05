from google.genai.errors import ServerError

from shared import gemini_retry


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
    def __init__(self, fail_times=0):
        self.models = FlakyModels(fail_times)


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
        gemini_retry.call_with_retry(client, "model", "prompt", max_retries=2)
        assert False, "expected ServerError"
    except ServerError:
        pass

    assert client.models.calls == 2 and waits == [2]  # one retry, one 2-second wait, then the caller can fail over


def test_the_default_is_still_four_attempts():
    assert gemini_retry.DEFAULT_MAX_RETRIES == 4
