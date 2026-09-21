import json

import classify_domain as cd


def test_slugify_title_produces_kebab_case():
    assert cd.slugify_title("Protecting Context and Prompts: Deterministic Security") == \
        "protecting-context-and-prompts-deterministic"


def test_slugify_title_handles_empty_text():
    assert cd.slugify_title("") == "unknown-domain"


def test_get_document_title_skips_front_matter_labels(tmp_path, monkeypatch):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    monkeypatch.setattr(cd, "TEXT_DIR", text_dir)

    doc = {
        "blocks": [
            {"label": "section_header", "text": "OPEN ACCESS", "page": 1},
            {"label": "section_header", "text": "REVIEWED BY", "page": 1},
            {"label": "section_header", "text": "Deep learning object detection-based early detection of lung cancer", "page": 1},
        ]
    }
    (text_dir / "sample.json").write_text(json.dumps(doc), encoding="utf-8")

    title = cd.get_document_title(text_dir / "sample.json")
    assert title == "Deep learning object detection-based early detection of lung cancer"


def test_get_document_title_falls_back_to_stem_when_no_headers(tmp_path, monkeypatch):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    monkeypatch.setattr(cd, "TEXT_DIR", text_dir)

    (text_dir / "sample.json").write_text(json.dumps({"blocks": []}), encoding="utf-8")

    title = cd.get_document_title(text_dir / "sample.json")
    assert title == "sample"


class FakeModel:
    def __init__(self, vector_map):
        self.vector_map = vector_map

    def encode(self, text_or_texts, normalize_embeddings=True):
        if isinstance(text_or_texts, list):
            return [self.vector_map[t] for t in text_or_texts]
        return self.vector_map[text_or_texts]


def test_classify_domain_assigns_seed_domain_when_similar(tmp_path, monkeypatch):
    text_embeddings_dir = tmp_path / "embeddings" / "text"
    text_dir = tmp_path / "text"
    text_embeddings_dir.mkdir(parents=True)
    text_dir.mkdir()

    monkeypatch.setattr(cd, "TEXT_EMBEDDINGS_DIR", text_embeddings_dir)
    monkeypatch.setattr(cd, "TEXT_DIR", text_dir)
    monkeypatch.setattr(cd, "OUTPUT_PATH", tmp_path / "domain_classification.json")
    monkeypatch.setattr(cd, "REPORT_PATH", tmp_path / "domain_classification_report.json")

    lung_label = cd.SEED_DOMAIN_LABELS["lung-cancer"]
    land_label = cd.SEED_DOMAIN_LABELS["land-cover"]

    doc = {
        "source_pdf": "sample.pdf",
        "records": [{"embedding": [1.0, 0.0, 0.0]}, {"embedding": [1.0, 0.0, 0.0]}],
    }
    (text_embeddings_dir / "sample.json").write_text(json.dumps(doc), encoding="utf-8")

    vector_map = {
        lung_label: [1.0, 0.0, 0.0],   # matches doc vector exactly
        land_label: [0.0, 1.0, 0.0],   # orthogonal, no match
    }
    monkeypatch.setattr(cd, "SentenceTransformer", lambda name: FakeModel(vector_map))

    cd.classify_domain()

    result = json.loads((tmp_path / "domain_classification.json").read_text(encoding="utf-8"))
    assert result["sample.pdf"] == "lung-cancer"


def test_classify_domain_creates_new_domain_when_no_seed_matches(tmp_path, monkeypatch):
    text_embeddings_dir = tmp_path / "embeddings" / "text"
    text_dir = tmp_path / "text"
    text_embeddings_dir.mkdir(parents=True)
    text_dir.mkdir()

    monkeypatch.setattr(cd, "TEXT_EMBEDDINGS_DIR", text_embeddings_dir)
    monkeypatch.setattr(cd, "TEXT_DIR", text_dir)
    monkeypatch.setattr(cd, "OUTPUT_PATH", tmp_path / "domain_classification.json")
    monkeypatch.setattr(cd, "REPORT_PATH", tmp_path / "domain_classification_report.json")

    lung_label = cd.SEED_DOMAIN_LABELS["lung-cancer"]
    land_label = cd.SEED_DOMAIN_LABELS["land-cover"]
    title = "AI Security Paper"

    doc = {
        "source_pdf": "outlier.pdf",
        "records": [{"embedding": [0.0, 0.0, 1.0]}],
    }
    (text_embeddings_dir / "outlier.json").write_text(json.dumps(doc), encoding="utf-8")
    (text_dir / "outlier.json").write_text(
        json.dumps({"blocks": [{"label": "section_header", "text": title, "page": 1}]}),
        encoding="utf-8",
    )

    vector_map = {
        lung_label: [1.0, 0.0, 0.0],
        land_label: [0.0, 1.0, 0.0],
        title: [0.0, 0.0, 1.0],
    }
    monkeypatch.setattr(cd, "SentenceTransformer", lambda name: FakeModel(vector_map))

    cd.classify_domain()

    result = json.loads((tmp_path / "domain_classification.json").read_text(encoding="utf-8"))
    assert result["outlier.pdf"] == cd.slugify_title(title)

    report = json.loads((tmp_path / "domain_classification_report.json").read_text(encoding="utf-8"))
    assert report["documents"][0]["created_new_domain"] is True
