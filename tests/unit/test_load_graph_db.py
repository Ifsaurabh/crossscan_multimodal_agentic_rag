import json

import load_graph_db as lgd


class FakeSession:
    def __init__(self):
        self.queries = []

    def run(self, query, **kwargs):
        self.queries.append((query, kwargs))
        return self


def test_load_papers_merges_each_with_domain():
    session = FakeSession()
    domain_map = {"a.pdf": "lung-cancer", "b.pdf": "land-cover"}

    count = lgd.load_papers(session, domain_map)

    assert count == 2
    assert len(session.queries) == 2
    assert session.queries[0][1]["source_pdf"] == "a.pdf"
    assert session.queries[0][1]["domain"] == "lung-cancer"


def test_load_entities_only_uses_verified_entities():
    session = FakeSession()
    entities = {
        "a.pdf": {
            "verified": {"methods": ["CNN"], "datasets": ["LIDC"], "metrics": ["accuracy"], "baselines": ["SVM"]},
            "unverified": {"methods": ["FakeMethod"], "datasets": [], "metrics": [], "baselines": ["FakeBaseline"]},
        }
    }

    method_count, dataset_count, metric_count, baseline_count = lgd.load_entities(session, entities)

    assert method_count == 1
    assert dataset_count == 1
    assert metric_count == 1
    assert baseline_count == 1
    all_names = [q[1].get("name") for q in session.queries]
    assert "FakeMethod" not in all_names and "FakeBaseline" not in all_names
    assert "CNN" in all_names and "SVM" in all_names


def test_load_sections_extracts_page_ranges(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    monkeypatch.setattr(lgd, "CHUNKS_DIR", chunks_dir)

    doc = {
        "source_pdf": "a.pdf",
        "parents": [
            {"chunk_id": "a_p1", "section": "Intro", "page_start": 1, "page_end": 2, "text": "x"},
            {"chunk_id": "a_p2", "section": "Results", "page_start": 3, "page_end": 3, "text": "y"},
        ],
    }
    (chunks_dir / "a.json").write_text(json.dumps(doc), encoding="utf-8")

    session = FakeSession()
    count, all_sections = lgd.load_sections(session)

    assert count == 2
    assert len(all_sections) == 2
    assert all_sections[0]["chunk_id"] == "a_p1"
    assert all_sections[1]["page_start"] == 3


def test_load_images_links_to_matching_section_by_page(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    metadata_path = images_dir / "metadata.json"
    monkeypatch.setattr(lgd, "IMAGES_METADATA_PATH", metadata_path)

    metadata = [
        {"image_file": "fig1.png", "source_pdf": "a.pdf", "page": 3},
        {"image_file": "fig2.png", "source_pdf": "a.pdf", "page": 99},  # no matching section
    ]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    all_sections = [
        {"chunk_id": "a_p1", "source_pdf": "a.pdf", "page_start": 1, "page_end": 2},
        {"chunk_id": "a_p2", "source_pdf": "a.pdf", "page_start": 3, "page_end": 3},
    ]

    session = FakeSession()
    image_count, linked_count = lgd.load_images(session, all_sections)

    assert image_count == 2
    assert linked_count == 1  # only fig1.png (page 3) matches a_p2's range

    near_section_queries = [q for q in session.queries if "NEAR_SECTION" in q[0]]
    assert len(near_section_queries) == 1
    assert near_section_queries[0][1]["chunk_id"] == "a_p2"


def test_load_images_handles_no_sections_for_paper(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    metadata_path = images_dir / "metadata.json"
    monkeypatch.setattr(lgd, "IMAGES_METADATA_PATH", metadata_path)

    metadata = [{"image_file": "fig1.png", "source_pdf": "unknown.pdf", "page": 1}]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    session = FakeSession()
    image_count, linked_count = lgd.load_images(session, [])

    assert image_count == 1
    assert linked_count == 0
