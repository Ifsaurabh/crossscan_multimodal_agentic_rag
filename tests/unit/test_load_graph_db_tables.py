"""load_graph_db: Table nodes (text stored on the node, linked to a Section by page overlap)
and Baseline nodes (COMPARED_TO). Uses a recording fake Neo4j session and temporary data files."""
import json

import pytest

import load_graph_db as lgd


class FakeSession:
    def __init__(self):
        self.runs = []

    def run(self, query, **params):
        self.runs.append((query, params))

    def find(self, needle):
        return [(q, p) for q, p in self.runs if needle in q]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeDriver:
    def __init__(self):
        self.session_obj = FakeSession()

    def session(self):
        return self.session_obj


def section(chunk_id, source_pdf="a.pdf", start=10, end=15):
    return {"chunk_id": chunk_id, "source_pdf": source_pdf, "page_start": start, "page_end": end}


def write_guarded(folder, file_name, source_pdf, tables):
    document = {"source_pdf": source_pdf}
    if tables is not None:
        document["tables"] = tables
    (folder / file_name).write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    monkeypatch.setattr(lgd, "GUARDED_DIR", tmp_path)
    return tmp_path


# ---------- load_tables: ids and stored content ----------

def test_table_ids_are_numbered_from_one_within_each_document(guarded):
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "t1", "page": 1}, {"text": "t2", "page": 2}])
    write_guarded(guarded, "b.json", "b.pdf", [{"text": "t3", "page": 1}])
    session = FakeSession()

    lgd.load_tables(session, [])

    ids = [params["table_id"] for _, params in session.find("MERGE (t:Table")]
    assert ids == ["a.pdf_table1", "a.pdf_table2", "b.pdf_table1"]


def test_the_table_text_and_page_are_stored_on_the_node_and_the_paper_is_linked(guarded):
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "Model  F1\nVGG16  0.98", "page": 14}])
    session = FakeSession()

    lgd.load_tables(session, [])

    query, params = session.find("MERGE (t:Table")[0]
    assert "MERGE (t:Table {table_id: $table_id})" in query
    assert "SET t.text = $text, t.page = $page" in query and "APPEARS_IN" in query
    assert params == {"table_id": "a.pdf_table1", "text": "Model  F1\nVGG16  0.98", "page": 14, "source_pdf": "a.pdf"}


def test_special_characters_in_a_table_are_stored_verbatim(guarded):
    text = "Métrique | Précision ± 0.5 | 98,7 % | \"quoted\" | 肺癌"
    write_guarded(guarded, "a.json", "a.pdf", [{"text": text, "page": 3}])
    session = FakeSession()

    lgd.load_tables(session, [])

    assert session.find("MERGE (t:Table")[0][1]["text"] == text


def test_reloading_gives_the_same_ids_so_the_load_is_repeatable(guarded):
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "t1", "page": 1}, {"text": "t2", "page": 2}])
    first, second = FakeSession(), FakeSession()

    lgd.load_tables(first, [])
    lgd.load_tables(second, [])

    assert [p["table_id"] for _, p in first.runs] == [p["table_id"] for _, p in second.runs]


# ---------- load_tables: linking a table to the section it sits in ----------

def test_a_table_inside_a_sections_page_range_is_linked_to_it(guarded):
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "t", "page": 12}])
    session = FakeSession()

    counts = lgd.load_tables(session, [section("a_p1")])

    query, params = session.find("NEAR_SECTION")[0]
    assert "MERGE (t)-[:NEAR_SECTION]->(s)" in query
    assert params == {"table_id": "a.pdf_table1", "chunk_id": "a_p1"}
    assert counts == (1, 1)


@pytest.mark.parametrize("page,linked", [(10, True), (15, True), (9, False), (16, False)])
def test_the_page_range_boundaries_are_inclusive(guarded, page, linked):
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "t", "page": page}])
    session = FakeSession()

    _, linked_count = lgd.load_tables(session, [section("a_p1", start=10, end=15)])

    assert linked_count == (1 if linked else 0)


def test_a_table_with_no_page_is_stored_but_never_linked(guarded):
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "t", "page": None}])
    session = FakeSession()

    counts = lgd.load_tables(session, [section("a_p1")])

    assert counts == (1, 0) and session.find("NEAR_SECTION") == []


def test_sections_without_page_information_are_skipped(guarded):
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "t", "page": 12}])
    session = FakeSession()

    counts = lgd.load_tables(session, [section("a_p0", start=None, end=None), section("a_p1")])

    assert counts == (1, 1) and session.find("NEAR_SECTION")[0][1]["chunk_id"] == "a_p1"


def test_when_two_sections_cover_the_page_the_first_wins(guarded):
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "t", "page": 12}])
    session = FakeSession()

    lgd.load_tables(session, [section("first", start=10, end=15), section("second", start=11, end=13)])

    assert [p["chunk_id"] for _, p in session.find("NEAR_SECTION")] == ["first"]


def test_only_sections_of_the_same_paper_are_considered(guarded):
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "t", "page": 12}])
    session = FakeSession()

    counts = lgd.load_tables(session, [section("b_p1", source_pdf="b.pdf")])

    assert counts == (1, 0)


def test_a_paper_with_no_sections_still_gets_its_tables(guarded):
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "t", "page": 12}])
    session = FakeSession()

    assert lgd.load_tables(session, []) == (1, 0)


# ---------- load_tables: empty and odd inputs ----------

def test_a_document_with_no_tables_key_or_an_empty_list_adds_nothing(guarded):
    write_guarded(guarded, "a.json", "a.pdf", None)
    write_guarded(guarded, "b.json", "b.pdf", [])
    session = FakeSession()

    assert lgd.load_tables(session, [section("a_p1")]) == (0, 0)
    assert session.runs == []


def test_an_empty_folder_loads_nothing(guarded):
    assert lgd.load_tables(FakeSession(), []) == (0, 0)


def test_documents_are_processed_in_file_name_order(guarded):
    write_guarded(guarded, "b.json", "b.pdf", [{"text": "t", "page": 1}])
    write_guarded(guarded, "a.json", "a.pdf", [{"text": "t", "page": 1}])
    session = FakeSession()

    lgd.load_tables(session, [])

    assert [p["source_pdf"] for _, p in session.runs] == ["a.pdf", "b.pdf"]


# ---------- load_entities: baselines ----------

def entities(**verified):
    return {"a.pdf": {"verified": verified}}


def test_baselines_are_loaded_as_compared_to_relationships():
    session = FakeSession()

    counts = lgd.load_entities(session, entities(baselines=["ResNet50", "VGG16"]))

    queries = session.find("MERGE (b:Baseline {name: $name})")
    assert [p["name"] for _, p in queries] == ["ResNet50", "VGG16"]
    assert all("MERGE (p)-[:COMPARED_TO]->(b)" in q for q, _ in queries)
    assert counts == (0, 0, 0, 2)


def test_load_entities_returns_counts_for_all_four_kinds():
    session = FakeSession()

    counts = lgd.load_entities(session, entities(
        methods=["CNN"], datasets=["LIDC-IDRI", "LUNA16"], metrics=["accuracy"], baselines=["SVM"],
    ))

    assert counts == (1, 2, 1, 1)


def test_an_entity_file_from_before_baselines_existed_still_loads():
    session = FakeSession()

    counts = lgd.load_entities(session, entities(methods=["CNN"], datasets=["LUNA16"], metrics=["F1"]))

    assert counts == (1, 1, 1, 0) and session.find("Baseline") == []


def test_baselines_are_counted_across_papers():
    session = FakeSession()
    data = {"a.pdf": {"verified": {"baselines": ["SVM"]}}, "b.pdf": {"verified": {"baselines": ["SVM", "RF"]}}}

    counts = lgd.load_entities(session, data)

    assert counts[3] == 3
    assert [p["source_pdf"] for _, p in session.find("Baseline")] == ["a.pdf", "b.pdf", "b.pdf"]


def test_a_paper_with_nothing_verified_adds_nothing():
    session = FakeSession()

    assert lgd.load_entities(session, entities()) == (0, 0, 0, 0)
    assert session.runs == []


# ---------- the whole load, from files, in order ----------

def test_the_full_graph_load_runs_every_step_in_order_and_releases_the_driver(tmp_path, monkeypatch, capsys):
    (tmp_path / "chunks").mkdir()
    (tmp_path / "guarded").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "domain.json").write_text(json.dumps([{"source_pdf": "a.pdf", "title": "A", "domain": "lung"}]), encoding="utf-8")
    (tmp_path / "entities.json").write_text(json.dumps({"a.pdf": {"verified": {
        "methods": ["CNN"], "datasets": [], "metrics": ["accuracy"], "baselines": ["SVM"]}}}), encoding="utf-8")
    (tmp_path / "chunks" / "a.json").write_text(json.dumps({"source_pdf": "a.pdf", "parents": [
        {"chunk_id": "a_p1", "section": "Results", "page_start": 10, "page_end": 15}]}), encoding="utf-8")
    (tmp_path / "images" / "metadata.json").write_text(json.dumps([
        {"image_file": "fig1.png", "source_pdf": "a.pdf", "page": 12}]), encoding="utf-8")
    write_guarded(tmp_path / "guarded", "a.json", "a.pdf", [{"text": "table text", "page": 12}])
    monkeypatch.setattr(lgd, "DOMAIN_MAP_PATH", tmp_path / "domain.json")
    monkeypatch.setattr(lgd, "ENTITIES_PATH", tmp_path / "entities.json")
    monkeypatch.setattr(lgd, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(lgd, "IMAGES_METADATA_PATH", tmp_path / "images" / "metadata.json")
    monkeypatch.setattr(lgd, "GUARDED_DIR", tmp_path / "guarded")
    driver = FakeDriver()
    released = []
    monkeypatch.setattr(lgd, "get_driver", lambda: driver)
    monkeypatch.setattr(lgd, "close_driver", lambda: released.append(True))

    lgd.load_graph_db()

    queries = [query for query, _ in driver.session_obj.runs]

    def first_position(marker):
        return next(i for i, query in enumerate(queries) if marker in query)

    steps = ["MERGE (p:Paper", "USES_METHOD", "COMPARED_TO", "MERGE (s:Section", "MERGE (i:Image", "MERGE (t:Table"]
    positions = [first_position(step) for step in steps]
    assert positions == sorted(positions)  # papers, entities (incl. baselines), sections, images, then tables
    assert released == [True]
    output = capsys.readouterr().out
    assert "Loaded 1 Paper nodes" in output
    assert "1 COMPARED_TO relationships" in output
    assert "Loaded 1 Table nodes, 1 linked" in output
    assert "Graph load complete." in output
