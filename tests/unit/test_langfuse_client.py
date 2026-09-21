import langfuse_client as lc


def test_disabled_flag_wins_over_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_DISABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert lc.is_enabled() is False


def test_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_DISABLED", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert lc.is_enabled() is False


def test_enabled_with_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_DISABLED", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert lc.is_enabled() is True


def test_get_client_and_callbacks_are_noops_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_DISABLED", "1")
    assert lc.get_client() is None
    assert lc.get_callbacks() == []
    lc.flush()  # must not raise
