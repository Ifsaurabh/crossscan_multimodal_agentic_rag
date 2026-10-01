"""ingestion_guardrails: injection is flag-only with a score (mocked here); redaction and
unsafe-keyword dropping are covered by test_ingestion_guardrails.py."""
import json

import ingestion_guardrails as ig
import query_guardrail as qg


def run(tmp_path, monkeypatch, doc, score):
    prepared, guarded = tmp_path / "prepared", tmp_path / "guarded"
    prepared.mkdir()
    monkeypatch.setattr(ig, "PREPARED_DIR", prepared)
    monkeypatch.setattr(ig, "GUARDED_DIR", guarded)
    monkeypatch.setattr(ig, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(ig, "DOMAIN_MAP_PATH", tmp_path / "no_domains.json")
    monkeypatch.setattr(qg, "injection_score", lambda text: score)
    (prepared / "doc.json").write_text(json.dumps(doc), encoding="utf-8")
    ig.run_guardrails()
    return (json.loads((guarded / "doc.json").read_text(encoding="utf-8")),
            json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["documents"][0])


DOC = {
    "source_pdf": "security.pdf",
    "sections": [{"heading": "Attacks", "text": "Attackers write: ignore previous instructions.", "page_start": 1, "page_end": 1}],
    "tables": [{"section_heading": "Attacks", "text": "payload | ignore previous instructions", "page_start": 1, "page_end": 1}],
}


def test_a_high_scoring_section_is_reported_with_its_score_but_never_dropped(tmp_path, monkeypatch):
    doc, entry = run(tmp_path, monkeypatch, json.loads(json.dumps(DOC)), 0.99)

    assert [s["heading"] for s in doc["sections"]] == ["Attacks"] and len(doc["tables"]) == 1
    assert [f["score"] for f in entry["injection_flags"]] == [0.99, 0.99]  # section + table
    assert entry["sections_dropped"] == [] and entry["tables_dropped"] == []


def test_a_low_scoring_section_raises_no_flag(tmp_path, monkeypatch):
    _, entry = run(tmp_path, monkeypatch, json.loads(json.dumps(DOC)), 0.1)
    assert entry["injection_flags"] == []


def test_when_the_model_is_unavailable_ingestion_still_redacts(tmp_path, monkeypatch):
    doc = {"source_pdf": "a.pdf", "sections": [{"heading": "Contact", "text": "Mail a@b.org", "page_start": 1, "page_end": 1}]}
    result, entry = run(tmp_path, monkeypatch, doc, None)
    assert result["sections"][0]["text"] == "Mail [REDACTED_EMAIL]"
    assert entry["injection_flags"] == [] and entry["pii_redactions"] == 1
