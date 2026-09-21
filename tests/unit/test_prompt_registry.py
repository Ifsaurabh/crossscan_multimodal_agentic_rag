import prompt_registry as pr


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


def test_local_prompts_covers_every_managed_prompt():
    prompts = pr.local_prompts()

    assert set(prompts) == {
        "transform-route", "quality-check", "generate-answer", "general-knowledge", "conversation-summary",
    }
    assert all(isinstance(text, str) and text for text in prompts.values())


def test_prompt_hash_is_deterministic_and_content_sensitive():
    assert pr.prompt_hash("abc") == pr.prompt_hash("abc")
    assert pr.prompt_hash("abc") != pr.prompt_hash("abd")
    assert len(pr.prompt_hash("abc")) == 12


def test_local_prompt_versions_one_hash_per_prompt():
    versions = pr.local_prompt_versions()
    assert set(versions) == set(pr.local_prompts())


def test_sync_prompts_creates_missing_and_skips_unchanged(monkeypatch):
    local = pr.local_prompts()
    unchanged_name = "quality-check"
    fake = FakeLangfuse(stored={unchanged_name: local[unchanged_name]})
    monkeypatch.setattr(pr.langfuse_client, "get_client", lambda: fake)

    results = pr.sync_prompts()

    assert results[unchanged_name].startswith("unchanged")
    created_names = {name for name, _, _ in fake.created}
    assert created_names == set(local) - {unchanged_name}
    assert all(labels == ["production"] for _, _, labels in fake.created)


def test_sync_prompts_creates_new_version_when_content_changed(monkeypatch):
    fake = FakeLangfuse(stored={"quality-check": "OLD TEXT"})
    monkeypatch.setattr(pr.langfuse_client, "get_client", lambda: fake)

    results = pr.sync_prompts()

    assert results["quality-check"].startswith("created")


def test_sync_prompts_requires_langfuse(monkeypatch):
    monkeypatch.setattr(pr.langfuse_client, "get_client", lambda: None)
    try:
        pr.sync_prompts()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
