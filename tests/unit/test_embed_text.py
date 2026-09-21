import json

import numpy as np

import embed_text as et
import embedding_config as cfg


class FakeModel:
    def __init__(self, seen_texts):
        self.seen_texts = seen_texts

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        self.seen_texts.extend(texts)
        return np.array([[float(len(t)), 0.0, 1.0] for t in texts])


def test_embed_text_prepends_structure_context_and_writes_output(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "chunks"
    embeddings_dir = tmp_path / "embeddings" / "text"
    chunks_dir.mkdir()

    monkeypatch.setattr(et, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(et, "EMBEDDINGS_DIR", embeddings_dir)
    monkeypatch.setattr(et, "REPORT_PATH", tmp_path / "text_embedding_report.json")

    doc = {
        "source_pdf": "sample.pdf",
        "parents": [
            {
                "chunk_id": "sample_p1",
                "children": [
                    {
                        "chunk_id": "sample_c1",
                        "parent_id": "sample_p1",
                        "source_pdf": "sample.pdf",
                        "section": "Results",
                        "page_start": 3,
                        "page_end": 3,
                        "text": "The model achieved 98% accuracy.",
                    }
                ],
            }
        ],
    }
    (chunks_dir / "sample.json").write_text(json.dumps(doc), encoding="utf-8")

    seen_texts = []
    monkeypatch.setattr(et, "SentenceTransformer", lambda name: FakeModel(seen_texts))

    et.embed_text()

    assert len(seen_texts) == 1
    assert seen_texts[0].startswith("Document: sample.pdf | Section: Results\n\n")
    assert "The model achieved 98% accuracy." in seen_texts[0]

    out_path = embeddings_dir / "sample.json"
    assert out_path.exists()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["embedding_version"] == cfg.EMBEDDING_VERSION
    assert len(result["records"]) == 1
    assert result["records"][0]["chunk_id"] == "sample_c1"
    assert result["records"][0]["parent_id"] == "sample_p1"
    assert "embedding" in result["records"][0]


def test_embed_text_skips_documents_with_no_children(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "chunks"
    embeddings_dir = tmp_path / "embeddings" / "text"
    chunks_dir.mkdir()

    monkeypatch.setattr(et, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(et, "EMBEDDINGS_DIR", embeddings_dir)
    monkeypatch.setattr(et, "REPORT_PATH", tmp_path / "text_embedding_report.json")

    doc = {"source_pdf": "empty.pdf", "parents": [{"chunk_id": "empty_p1", "children": []}]}
    (chunks_dir / "empty.json").write_text(json.dumps(doc), encoding="utf-8")

    monkeypatch.setattr(et, "SentenceTransformer", lambda name: FakeModel([]))

    et.embed_text()

    assert not (embeddings_dir / "empty.json").exists()
