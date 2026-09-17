import json

import prepare_documents as pd


def test_is_references_heading_matches_common_variants():
    assert pd.is_references_heading("References")
    assert pd.is_references_heading("REFERENCES")
    assert pd.is_references_heading("6. REFERENCES")
    assert pd.is_references_heading("2.1 References")
    assert pd.is_references_heading("Bibliography")
    assert pd.is_references_heading("Reference List:")


def test_is_references_heading_rejects_non_matches():
    assert not pd.is_references_heading("Introduction")
    assert not pd.is_references_heading("1. Methodology")
    assert not pd.is_references_heading(None)
    assert not pd.is_references_heading("")


def test_split_into_sections_groups_blocks_and_tracks_pages():
    blocks = [
        {"label": "text", "level": None, "page": 1, "text": "preamble text"},
        {"label": "section_header", "level": 1, "page": 1, "text": "Abstract"},
        {"label": "text", "level": None, "page": 1, "text": "abstract body"},
        {"label": "section_header", "level": 1, "page": 2, "text": "Introduction"},
        {"label": "text", "level": None, "page": 2, "text": "intro part one"},
        {"label": "text", "level": None, "page": 3, "text": "intro part two"},
    ]

    sections = pd.split_into_sections(blocks)

    assert len(sections) == 3

    preamble = sections[0]
    assert preamble["heading"] is None
    assert preamble["page_start"] == 1
    assert preamble["page_end"] == 1

    abstract = sections[1]
    assert abstract["heading"] == "Abstract"
    assert "abstract body" in abstract["text"]
    assert abstract["page_start"] == 1
    assert abstract["page_end"] == 1

    intro = sections[2]
    assert intro["heading"] == "Introduction"
    assert "intro part one" in intro["text"]
    assert "intro part two" in intro["text"]
    assert intro["page_start"] == 2
    assert intro["page_end"] == 3


def test_prepare_documents_drops_references_and_writes_output(tmp_path, monkeypatch):
    text_dir = tmp_path / "text"
    prepared_dir = tmp_path / "prepared"
    text_dir.mkdir()

    monkeypatch.setattr(pd, "TEXT_DIR", text_dir)
    monkeypatch.setattr(pd, "PREPARED_DIR", prepared_dir)

    doc = {
        "source_pdf": "sample.pdf",
        "blocks": [
            {"label": "section_header", "level": 1, "page": 1, "text": "Introduction"},
            {"label": "text", "level": None, "page": 1, "text": "Some intro content."},
            {"label": "section_header", "level": 1, "page": 5, "text": "References"},
            {"label": "text", "level": None, "page": 5, "text": "[1] Some citation."},
        ],
    }
    (text_dir / "sample.json").write_text(json.dumps(doc), encoding="utf-8")

    pd.prepare_documents()

    out_path = prepared_dir / "sample.json"
    assert out_path.exists()

    result = json.loads(out_path.read_text(encoding="utf-8"))
    headings = [s["heading"] for s in result["sections"]]
    assert "References" not in headings
    assert "Introduction" in headings
