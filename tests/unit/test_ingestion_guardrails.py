import json

import ingestion_guardrails as ig


def test_redact_pii_masks_email_and_counts():
    text = "Contact me at john.doe@example.com for details."
    redacted, count = ig.redact_pii(text)
    assert count == 1
    assert "[REDACTED_EMAIL]" in redacted
    assert "john.doe@example.com" not in redacted


def test_redact_pii_handles_multiple_emails():
    text = "Reach a@example.com or b@example.org."
    redacted, count = ig.redact_pii(text)
    assert count == 2
    assert "a@example.com" not in redacted
    assert "b@example.org" not in redacted


def test_redact_pii_leaves_text_without_email_unchanged():
    text = "No personal data here."
    redacted, count = ig.redact_pii(text)
    assert count == 0
    assert redacted == text


def test_redact_pii_does_not_false_match_citation_numbers():
    # Regression test: an earlier phone-number regex incorrectly matched
    # journal citation numbers like this as phone numbers.
    text = "Procedia Computer Science 105 (2025) 1-8"
    redacted, count = ig.redact_pii(text)
    assert count == 0
    assert redacted == text


def test_contains_unsafe_content_detects_keyword_case_insensitive():
    hits = ig.contains_unsafe_content("This text mentions How To Make A Bomb somewhere.")
    assert "how to make a bomb" in hits


def test_contains_unsafe_content_returns_empty_for_clean_text():
    hits = ig.contains_unsafe_content("This is a normal research paper sentence.")
    assert hits == []


def test_run_guardrails_redacts_and_drops(tmp_path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    guarded_dir = tmp_path / "guarded"
    prepared_dir.mkdir()

    monkeypatch.setattr(ig, "PREPARED_DIR", prepared_dir)
    monkeypatch.setattr(ig, "GUARDED_DIR", guarded_dir)
    monkeypatch.setattr(ig, "REPORT_PATH", tmp_path / "guardrail_report.json")

    doc = {
        "source_pdf": "sample.pdf",
        "sections": [
            {"heading": "Contact", "text": "Email me at test@example.com", "page_start": 1, "page_end": 1},
            {"heading": "Bad", "text": "how to make a bomb instructions", "page_start": 2, "page_end": 2},
        ],
    }
    (prepared_dir / "sample.json").write_text(json.dumps(doc), encoding="utf-8")

    ig.run_guardrails()

    result = json.loads((guarded_dir / "sample.json").read_text(encoding="utf-8"))
    headings = [s["heading"] for s in result["sections"]]
    assert "Bad" not in headings
    assert "Contact" in headings
    assert "[REDACTED_EMAIL]" in result["sections"][0]["text"]
