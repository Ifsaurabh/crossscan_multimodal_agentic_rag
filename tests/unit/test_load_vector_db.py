import json

import load_vector_db as lvd


class FakeConnection:
    def __init__(self):
        self.inserts = []
        self.committed = False
        self.closed = False

    def execute(self, sql, params=None):
        table = "text_parents" if "text_parents" in sql else \
            "text_chunks" if "text_chunks" in sql else \
            "images" if "images" in sql else "unknown"
        self.inserts.append((table, params))
        return self

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def test_load_text_inserts_parents_and_chunks_with_domain(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "chunks"
    text_embeddings_dir = tmp_path / "embeddings" / "text"
    chunks_dir.mkdir()
    text_embeddings_dir.mkdir(parents=True)

    monkeypatch.setattr(lvd, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(lvd, "TEXT_EMBEDDINGS_DIR", text_embeddings_dir)

    chunk_doc = {
        "source_pdf": "sample.pdf",
        "parents": [
            {
                "chunk_id": "sample_p1",
                "section": "Results",
                "page_start": 3,
                "page_end": 3,
                "text": "Full parent text.",
                "children": [
                    {
                        "chunk_id": "sample_c1",
                        "parent_id": "sample_p1",
                        "section": "Results",
                        "page_start": 3,
                        "page_end": 3,
                        "text": "Child text.",
                    }
                ],
            }
        ],
    }
    (chunks_dir / "sample.json").write_text(json.dumps(chunk_doc), encoding="utf-8")

    embedding_doc = {
        "records": [{"chunk_id": "sample_c1", "embedding": [0.1, 0.2, 0.3]}],
    }
    (text_embeddings_dir / "sample.json").write_text(json.dumps(embedding_doc), encoding="utf-8")

    fake_conn = FakeConnection()
    domain_map = {"sample.pdf": "lung-cancer"}

    parent_count, chunk_count = lvd.load_text(fake_conn, domain_map)

    assert parent_count == 1
    assert chunk_count == 1

    parent_inserts = [p for table, p in fake_conn.inserts if table == "text_parents"]
    chunk_inserts = [p for table, p in fake_conn.inserts if table == "text_chunks"]

    assert parent_inserts[0][5] == "lung-cancer"  # domain field
    assert chunk_inserts[0][6] == "lung-cancer"  # domain field
    assert chunk_inserts[0][9] == [0.1, 0.2, 0.3]  # embedding field


def test_load_text_skips_chunks_missing_embeddings(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "chunks"
    text_embeddings_dir = tmp_path / "embeddings" / "text"
    chunks_dir.mkdir()
    text_embeddings_dir.mkdir(parents=True)

    monkeypatch.setattr(lvd, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(lvd, "TEXT_EMBEDDINGS_DIR", text_embeddings_dir)

    chunk_doc = {
        "source_pdf": "sample.pdf",
        "parents": [{
            "chunk_id": "sample_p1", "section": "Intro", "page_start": 1, "page_end": 1,
            "text": "text", "children": [{
                "chunk_id": "missing_embedding_chunk", "parent_id": "sample_p1",
                "section": "Intro", "page_start": 1, "page_end": 1, "text": "child text",
            }],
        }],
    }
    (chunks_dir / "sample.json").write_text(json.dumps(chunk_doc), encoding="utf-8")
    # No matching embeddings file written at all

    fake_conn = FakeConnection()
    parent_count, chunk_count = lvd.load_text(fake_conn, {"sample.pdf": "land-cover"})

    assert parent_count == 1
    assert chunk_count == 0


def test_load_images_tags_domain_correctly(tmp_path, monkeypatch):
    image_embeddings_path = tmp_path / "image_embeddings.json"
    monkeypatch.setattr(lvd, "IMAGE_EMBEDDINGS_PATH", image_embeddings_path)

    data = {"records": [{"image_file": "fig1.png", "source_pdf": "sample.pdf", "page": 2, "embedding": [0.5, 0.6]}]}
    image_embeddings_path.write_text(json.dumps(data), encoding="utf-8")

    fake_conn = FakeConnection()
    count = lvd.load_images(fake_conn, {"sample.pdf": "land-cover"})

    assert count == 1
    image_inserts = [p for table, p in fake_conn.inserts if table == "images"]
    assert image_inserts[0][3] == "land-cover"  # domain field


def test_load_images_uses_unclassified_when_domain_unknown(tmp_path, monkeypatch):
    image_embeddings_path = tmp_path / "image_embeddings.json"
    monkeypatch.setattr(lvd, "IMAGE_EMBEDDINGS_PATH", image_embeddings_path)

    data = {"records": [{"image_file": "fig1.png", "source_pdf": "unknown.pdf", "page": 1, "embedding": [0.1]}]}
    image_embeddings_path.write_text(json.dumps(data), encoding="utf-8")

    fake_conn = FakeConnection()
    lvd.load_images(fake_conn, {})

    image_inserts = [p for table, p in fake_conn.inserts if table == "images"]
    assert image_inserts[0][3] == "unclassified"
