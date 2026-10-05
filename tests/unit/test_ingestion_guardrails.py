import json

from ingestion import ingestion_guardrails as ig


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


def test_guard_document_redacts_and_drops(monkeypatch):
    monkeypatch.setattr(ig.qg, "injection_scores", lambda texts, **k: [0.0] * len(texts))
    doc = {
        "source_pdf": "sample.pdf",
        "sections": [
            {"heading": "Contact", "text": "Email me at test@example.com", "page_start": 1, "page_end": 1},
            {"heading": "Bad", "text": "how to make a bomb instructions", "page_start": 2, "page_end": 2},
        ],
    }

    result, entry = ig.guard_document(doc)

    headings = [s["heading"] for s in result["sections"]]
    assert "Bad" not in headings
    assert "Contact" in headings
    assert "[REDACTED_EMAIL]" in result["sections"][0]["text"]
    assert entry["pii_redactions"] == 1 and entry["sections_dropped"][0]["heading"] == "Bad"


# ---------- a table's caption ----------

def guarded_table(caption, text="| a | b |\n|---|---|\n| 1 | 2 |"):
    doc = {"source_pdf": "x.pdf", "sections": [{"heading": "Results", "text": "Plain text.", "page_start": 1, "page_end": 1}],
           "tables": [{"section_heading": "Results", "text": text, "caption": caption, "page": 1}]}
    return ig.guard_document(doc)


def test_an_email_in_a_table_caption_is_redacted_and_counted(monkeypatch):
    monkeypatch.setattr(ig.qg, "injection_scores", lambda texts, **k: [0.0] * len(texts))
    doc, entry = guarded_table("Table 1: data from jane@example.com")

    assert "jane@example.com" not in doc["tables"][0]["caption"] and "[REDACTED_EMAIL]" in doc["tables"][0]["caption"]
    assert entry["pii_redactions"] == 1 and entry["pii_locations"][0]["where"] == "Results"


def test_a_table_with_an_unsafe_caption_is_dropped(monkeypatch):
    monkeypatch.setattr(ig.qg, "injection_scores", lambda texts, **k: [0.0] * len(texts))
    doc, entry = guarded_table("Table 1: how to make a bomb")

    assert doc["tables"] == [] and entry["tables_dropped"][0]["matched_keywords"] == ["how to make a bomb"]


def test_a_table_without_a_caption_is_kept_with_an_empty_one(monkeypatch):
    monkeypatch.setattr(ig.qg, "injection_scores", lambda texts, **k: [0.0] * len(texts))
    doc, _ = guarded_table("")
    assert doc["tables"][0]["caption"] == ""
