import json

import extract_entities as ee


def test_normalize_heading_strips_numbering():
    assert ee.normalize_heading("4. Conclusions") == "conclusions"
    assert ee.normalize_heading("1.2 Abstract") == "abstract"
    assert ee.normalize_heading(None) == ""


def test_get_abstract_and_conclusion_finds_both():
    sections = [
        {"heading": "Introduction", "text": "intro text"},
        {"heading": "Abstract", "text": "abstract text"},
        {"heading": "4. Conclusions", "text": "conclusion text"},
    ]
    abstract, conclusion = ee.get_abstract_and_conclusion(sections)
    assert abstract == "abstract text"
    assert conclusion == "conclusion text"


def test_get_abstract_and_conclusion_missing_returns_empty():
    sections = [{"heading": "Introduction", "text": "intro text"}]
    abstract, conclusion = ee.get_abstract_and_conclusion(sections)
    assert abstract == ""
    assert conclusion == ""


def test_parse_entities_handles_plain_json():
    result = ee.parse_entities('{"methods": ["CNN"], "datasets": [], "metrics": []}')
    assert result == {"methods": ["CNN"], "datasets": [], "metrics": []}


def test_parse_entities_strips_markdown_code_fence():
    raw = '```json\n{"methods": ["CNN"], "datasets": [], "metrics": []}\n```'
    result = ee.parse_entities(raw)
    assert result == {"methods": ["CNN"], "datasets": [], "metrics": []}


def test_parse_entities_extracts_json_from_surrounding_text():
    raw = 'Here is the result: {"methods": ["CNN"], "datasets": [], "metrics": []} Hope this helps!'
    result = ee.parse_entities(raw)
    assert result == {"methods": ["CNN"], "datasets": [], "metrics": []}


def test_parse_entities_returns_none_for_unparseable_text():
    assert ee.parse_entities("not json at all") is None


def test_verify_entities_splits_verified_and_unverified():
    entities = {"methods": ["CNN", "FakeMethod"], "datasets": [], "metrics": []}
    source_text = "This paper uses CNN for classification."

    verified, unverified = ee.verify_entities(entities, source_text)

    assert verified["methods"] == ["CNN"]
    assert unverified["methods"] == ["FakeMethod"]


def test_verify_entities_case_insensitive():
    entities = {"methods": ["cnn"], "datasets": [], "metrics": []}
    source_text = "This paper uses CNN for classification."

    verified, unverified = ee.verify_entities(entities, source_text)

    assert verified["methods"] == ["cnn"]
    assert unverified["methods"] == []


class FakeGeminiResponse:
    def __init__(self, text):
        self.text = text


class FakeGeminiModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = 0

    def generate_content(self, model, contents):
        self.calls += 1
        return FakeGeminiResponse(self.response_text)


class FakeGeminiClient:
    def __init__(self, response_text='{"methods": ["CNN"], "datasets": [], "metrics": []}'):
        self.models = FakeGeminiModels(response_text)


def test_extract_entities_processes_documents_and_skips_missing_abstract(tmp_path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    monkeypatch.setattr(ee, "PREPARED_DIR", prepared_dir)
    monkeypatch.setattr(ee, "OUTPUT_PATH", tmp_path / "entities.json")
    monkeypatch.setattr(ee, "REPORT_PATH", tmp_path / "entity_extraction_report.json")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    good_doc = {
        "source_pdf": "good.pdf",
        "sections": [{"heading": "Abstract", "text": "This paper uses CNN."}],
    }
    (prepared_dir / "good.json").write_text(json.dumps(good_doc), encoding="utf-8")

    no_abstract_doc = {
        "source_pdf": "noabstract.pdf",
        "sections": [{"heading": "Introduction", "text": "no abstract here"}],
    }
    (prepared_dir / "noabstract.json").write_text(json.dumps(no_abstract_doc), encoding="utf-8")

    monkeypatch.setattr(ee.genai, "Client", lambda api_key: FakeGeminiClient())

    ee.extract_entities()

    results = json.loads((tmp_path / "entities.json").read_text(encoding="utf-8"))
    assert "good.pdf" in results
    assert results["good.pdf"]["verified"]["methods"] == ["CNN"]
    assert "noabstract.pdf" not in results

    report = json.loads((tmp_path / "entity_extraction_report.json").read_text(encoding="utf-8"))
    skipped = [d for d in report["documents"] if d.get("skipped")]
    assert len(skipped) == 1
    assert skipped[0]["source_pdf"] == "noabstract.pdf"


def test_extract_entities_resume_skips_already_processed(tmp_path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    monkeypatch.setattr(ee, "PREPARED_DIR", prepared_dir)
    monkeypatch.setattr(ee, "OUTPUT_PATH", tmp_path / "entities.json")
    monkeypatch.setattr(ee, "REPORT_PATH", tmp_path / "entity_extraction_report.json")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    doc = {"source_pdf": "a.pdf", "sections": [{"heading": "Abstract", "text": "uses CNN"}]}
    (prepared_dir / "a.json").write_text(json.dumps(doc), encoding="utf-8")

    # Pre-existing report from the SAME model - should be treated as resumable.
    existing_report = {"model": ee.GEMINI_MODEL, "documents": [{"source_pdf": "a.pdf", "methods_count": 1}]}
    (tmp_path / "entity_extraction_report.json").write_text(json.dumps(existing_report), encoding="utf-8")
    (tmp_path / "entities.json").write_text(json.dumps({"a.pdf": {"verified": {"methods": ["CNN"]}, "unverified": {}}}), encoding="utf-8")

    fake_client = FakeGeminiClient()
    monkeypatch.setattr(ee.genai, "Client", lambda api_key: fake_client)

    ee.extract_entities()

    assert fake_client.models.calls == 0  # should not have called Gemini again


def test_extract_entities_ignores_stale_report_from_different_model(tmp_path, monkeypatch):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    monkeypatch.setattr(ee, "PREPARED_DIR", prepared_dir)
    monkeypatch.setattr(ee, "OUTPUT_PATH", tmp_path / "entities.json")
    monkeypatch.setattr(ee, "REPORT_PATH", tmp_path / "entity_extraction_report.json")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    doc = {"source_pdf": "a.pdf", "sections": [{"heading": "Abstract", "text": "uses CNN"}]}
    (prepared_dir / "a.json").write_text(json.dumps(doc), encoding="utf-8")

    # Stale report from a DIFFERENT model (e.g. the old Qwen run) - must NOT be trusted.
    stale_report = {"model": "qwen2.5:3b", "documents": [{"source_pdf": "a.pdf"}]}
    (tmp_path / "entity_extraction_report.json").write_text(json.dumps(stale_report), encoding="utf-8")
    (tmp_path / "entities.json").write_text(json.dumps({"a.pdf": {"methods": ["stale"]}}), encoding="utf-8")

    fake_client = FakeGeminiClient()
    monkeypatch.setattr(ee.genai, "Client", lambda api_key: fake_client)

    ee.extract_entities()

    assert fake_client.models.calls == 1  # re-processed despite stale report existing
