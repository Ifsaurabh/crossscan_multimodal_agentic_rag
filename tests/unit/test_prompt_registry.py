from shared import prompt_registry as pr


class FakePrompt:
    def __init__(self, text, version=1):
        self.prompt = text
        self.version = version


class FakeLangfuse:
    def __init__(self, stored=None, raise_on_get=False):
        self.stored = stored or {}
        self.raise_on_get = raise_on_get
        self.created = []

    def get_prompt(self, name, **kwargs):
        if self.raise_on_get or name not in self.stored:
            raise RuntimeError("no such prompt")
        return FakePrompt(self.stored[name])

    def create_prompt(self, name, prompt, labels, type):
        self.created.append((name, prompt, labels))
        return FakePrompt(prompt, version=len(self.created))


def test_get_prompt_falls_back_when_langfuse_disabled(monkeypatch):
    monkeypatch.setattr(pr.langfuse_client, "get_client", lambda: None)
    assert pr.get_prompt("transform-route", "LOCAL") == "LOCAL"


def test_get_prompt_returns_langfuse_version_when_available(monkeypatch):
    fake = FakeLangfuse(stored={"transform-route": "REMOTE"})
    monkeypatch.setattr(pr.langfuse_client, "get_client", lambda: fake)
    assert pr.get_prompt("transform-route", "LOCAL") == "REMOTE"


def test_get_prompt_falls_back_on_error(monkeypatch):
    monkeypatch.setattr(pr.langfuse_client, "get_client", lambda: FakeLangfuse(raise_on_get=True))
    assert pr.get_prompt("transform-route", "LOCAL") == "LOCAL"
