"""image_check: the exit logic of the worker image's self-check (the checks themselves need the real models)."""
import pytest

from ingestion import image_check as ic


@pytest.fixture
def fake_checks(monkeypatch):
    def install(**results):
        for name in ("tesseract", "tiktoken", "text_model", "image_model", "redaction", "prompt_guard", "extraction"):
            outcome = results.get(name, "fine")
            def check(outcome=outcome):
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
            monkeypatch.setattr(ic, f"check_{name}", check)
    return install


def test_everything_loading_passes(fake_checks, capsys):
    fake_checks()
    assert ic.run(require_prompt_guard=True, tesseract=True) is True
    assert capsys.readouterr().out.count("OK ") == 7


def test_a_missing_required_component_fails_the_check(fake_checks, capsys):
    fake_checks(text_model=RuntimeError("no model"))
    assert ic.run() is False
    assert "FAILED" in capsys.readouterr().out


def test_prompt_guard_is_only_a_warning_unless_required(fake_checks, capsys):
    fake_checks(prompt_guard=RuntimeError("no token"))
    assert ic.run() is True and "WARNING" in capsys.readouterr().out
    assert ic.run(require_prompt_guard=True) is False


def test_tesseract_is_only_checked_when_asked(fake_checks, capsys):
    fake_checks(tesseract=RuntimeError("not installed"))
    assert ic.run() is True
    assert ic.run(tesseract=True) is False


def test_the_exit_code_follows_the_result(fake_checks):
    fake_checks(redaction=RuntimeError("x"))
    with pytest.raises(SystemExit) as stopped:
        ic.main([])
    assert stopped.value.code == 1
