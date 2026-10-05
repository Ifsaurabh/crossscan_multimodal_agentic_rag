"""The ingestion pipeline's database work against a REAL Postgres (throwaway schema): replacing a document,
the domain registry (the domain is the upload folder), the cache invalidation and the manifest. Models, bucket and extraction are fakes.

Run:  python -m pytest tests/integration/test_pipeline_integration.py --run-integration
Nothing outside the throwaway `it_...` schema is touched (see conftest.py)."""
from contextlib import contextmanager

import numpy as np
import pytest

from ingestion import (document_loader, domain_registry, extraction, ingestion_manifest, ingestion_reports,
                       pipeline_schema)
from ingestion import intake_check as ic
from ingestion import pipeline as pl
from ingestion import remove_document
from retrieval import query_cache
from shared import db, setup_vector_db

pytestmark = pytest.mark.integration

DIM = 768


def vec(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v.tolist()


@pytest.fixture
def tables(schema, monkeypatch):
    assert schema.startswith("it_"), "must only run in the throwaway schema"
    for module in (document_loader, domain_registry, ingestion_manifest, ingestion_reports, remove_document, query_cache):
        monkeypatch.setattr(module, "SCHEMA_NAME", schema)
    conn = db.get_connection()
    for sql in (setup_vector_db.schema_sql(schema), pipeline_schema.setup_sql(schema), query_cache.setup_sql(schema),
                ingestion_reports.setup_sql(schema)):
        conn.execute(sql)
    conn.commit()
    for t in ("text_chunks", "text_parents", "images", "doc_tables", "domain_members", "domains", "query_cache",
              "ingestion_manifest", "ingestion_report_stages", "ingestion_reports"):
        conn.execute(f"TRUNCATE {schema}.{t} CASCADE")
    conn.commit()
    conn.close()
    return schema


def count(schema, table, where="TRUE", params=()):
    with db.get_connection() as c:
        return c.execute(f"SELECT COUNT(*) FROM {schema}.{table} WHERE {where}", params).fetchone()[0]


class Storage:
    def __init__(self):
        self.objects = {}

    def upload_bytes(self, name, data, content_type, cache_control=None):
        self.objects[name] = data

    def delete_prefix(self, prefix, keep=()):
        for name in [n for n in self.objects if n.startswith(prefix) and n not in set(keep)]:
            del self.objects[name]


NAME = "lung-cancer/paper.pdf"


def make_pipeline(storage, text, images=(), table_text=None, name=NAME, caption="Table 1: scores"):
    @contextmanager
    def connect():
        conn = db.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    blocks = [{"label": "section_header", "level": 1, "page": 1, "text": "Intro"},
              {"label": "text", "level": None, "page": 1, "text": text * 60}]
    if table_text:
        blocks.append({"label": "table", "level": None, "page": 1, "text": table_text, "caption": caption})
    result = extraction.ExtractionResult(source_name=name, blocks=blocks, pages=1, page_chars=[2000])
    return pl.Pipeline(storage, connect=connect, text_embedder=lambda texts: [vec(0) for _ in texts],
                       image_embedder=lambda data: [0.1] * 512, extract=lambda p, i: result,
                       image_lister=lambda p: list(images), guard_token_counter=lambda text: len(text.split()),
                       text_counter=lambda texts: (len(texts) * 7, 0), emit=lambda line: None)


def intake(h="a" * 64, status="new", name=NAME):
    return ic.IntakeResult(file_name=name, outcome=ic.ACCEPT, content_hash=h, version_status=status,
                           file_type="pdf", pages=1, document_label="text")


def test_a_document_is_stored_with_its_domain_images_and_manifest(tables):
    storage = Storage()
    pipeline = make_pipeline(storage, "lung nodule ", images=[{"page": 1, "filename": "page1_img1.png", "ext": "png",
                                                               "bytes": b"i"}], table_text="| a | b |")
    report = pipeline("paper.pdf", intake())

    assert report.domain == "lung-cancer" and report.domain_created  # a new folder is a new domain
    assert count(tables, "text_chunks", "source_pdf = %s AND domain = %s", (NAME, "lung-cancer")) == report.children > 0
    assert count(tables, "text_parents") == report.parents
    assert count(tables, "images") == 1 and "images/lung-cancer/paper/page1_img1.png" in storage.objects
    assert count(tables, "doc_tables") == 1
    with db.get_connection() as c:
        parent, caption = c.execute(f"SELECT parent_id, caption FROM {tables}.doc_tables").fetchone()
        assert parent and parent.startswith("lung-cancer/paper_p") and caption == "Table 1: scores"
        # the image sits on page 1: it is linked to the first parent that holds that page
        assert c.execute(f"SELECT parent_id FROM {tables}.images").fetchone()[0] == "lung-cancer/paper_p1"
        assert ingestion_manifest.get_active_hash(c, NAME) == "a" * 64
        assert c.execute(f"SELECT doc_count FROM {tables}.domains WHERE name = 'lung-cancer'").fetchone()[0] == 1


def test_a_changed_version_replaces_the_old_rows_and_the_whole_answer_cache_is_cleared(tables):
    storage = Storage()
    make_pipeline(storage, "old text ", images=[{"page": 1, "filename": "old.png", "ext": "png", "bytes": b"o"}])(
        "paper.pdf", intake("a" * 64))
    with db.get_connection() as c:
        c.execute(f"INSERT INTO {tables}.query_cache (query_hash, query_text, chunks_retrieved, answer) VALUES "
                  "('h1', 'q', %s::jsonb, 'a'), ('h2', 'q2', %s::jsonb, 'b')",
                  (f'[{{"source_pdf": "{NAME}"}}]', '[{"source_pdf": "other/other.pdf"}]'))
        c.commit()

    report = make_pipeline(storage, "new text ", images=[{"page": 1, "filename": "new.png", "ext": "png", "bytes": b"n"}])(
        "paper.pdf", intake("b" * 64, "changed"))

    # a cached answer built before this document changed may be improved by it: the whole cache goes
    assert report.cache_entries_removed == 2 and count(tables, "query_cache") == 0
    assert count(tables, "text_chunks", "text LIKE %s", ("%old text%",)) == 0
    assert count(tables, "text_chunks") == report.children and count(tables, "text_parents") == report.parents
    assert set(storage.objects) == {"images/lung-cancer/paper/new.png"}
    with db.get_connection() as c:
        assert ingestion_manifest.get_active_hash(c, NAME) == "b" * 64
        assert c.execute(f"SELECT doc_count FROM {tables}.domains").fetchall() == [(1,)]  # not counted twice


def test_a_second_document_in_the_same_folder_joins_the_domain(tables):
    storage = Storage()
    make_pipeline(storage, "lung ")("paper.pdf", intake())
    two = "lung-cancer/two.pdf"

    report = make_pipeline(storage, "lung ", name=two)("two.pdf", intake("c" * 64, name=two))

    assert report.domain == "lung-cancer" and not report.domain_created
    assert count(tables, "domains") == 1
    with db.get_connection() as c:
        assert c.execute(f"SELECT doc_count FROM {tables}.domains").fetchone()[0] == 2


def test_the_same_file_name_in_two_domains_never_meets(tables):
    storage = Storage()
    make_pipeline(storage, "lung ")("paper.pdf", intake())
    other = "land-cover/paper.pdf"

    report = make_pipeline(storage, "satellite ", name=other)("paper.pdf", intake("d" * 64, name=other))

    assert report.domain == "land-cover" and report.domain_created and count(tables, "domains") == 2
    assert count(tables, "text_chunks", "source_pdf = %s", (NAME,)) > 0
    assert count(tables, "text_chunks", "source_pdf = %s", (other,)) > 0
    assert count(tables, "text_parents", "parent_id LIKE %s", ("lung-cancer/%",)) > 0
    assert count(tables, "text_parents", "parent_id LIKE %s", ("land-cover/%",)) > 0  # chunk ids do not collide
    with db.get_connection() as c:
        assert ingestion_manifest.get_active_hash(c, NAME) == "a" * 64
        assert ingestion_manifest.get_active_hash(c, other) == "d" * 64


def test_an_error_during_loading_leaves_nothing_behind(tables, monkeypatch):
    def boom(conn, *a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(ingestion_manifest, "mark_ingested", boom)
    with pytest.raises(RuntimeError):
        make_pipeline(Storage(), "lung ")("paper.pdf", intake())

    for table in ("text_chunks", "text_parents", "domain_members", "domains", "ingestion_manifest"):
        assert count(tables, table) == 0


def test_every_attempt_is_recorded_with_a_growing_report_version_and_its_stages(tables):
    storage = Storage()
    make_pipeline(storage, "old text ")("paper.pdf", intake("a" * 64))
    make_pipeline(storage, "new text ")("paper.pdf", intake("b" * 64, "changed"))

    with db.get_connection() as c:
        rows = c.execute(f"SELECT report_version, status, domain, content_hash, embedding_version, pipeline_version, "
                         f"total_seconds, counts FROM {tables}.ingestion_reports WHERE source_pdf = %s ORDER BY report_version",
                         (NAME,)).fetchall()
        assert [r[0] for r in rows] == [1, 2] and [r[1] for r in rows] == ["ingested", "ingested"]
        assert rows[0][2] == "lung-cancer" and rows[0][3] == "a" * 64 and rows[1][3] == "b" * 64
        assert rows[0][4] and rows[0][5] == "p1" and rows[0][6] is not None
        assert rows[1][7]["children"] > 0 and rows[1][7]["pages"] == 1
        stages = ingestion_reports.stage_rows(c, c.execute(f"SELECT MAX(report_id) FROM {tables}.ingestion_reports").fetchone()[0])
    assert [s["stage"] for s in stages] == ["extract", "quality", "prepare", "guardrails", "chunk", "embed_text", "images",
                                            "domain", "replace", "manifest_commit"]
    by_stage = {s["stage"]: s for s in stages}
    assert by_stage["guardrails"]["token_kind"] == "prompt_guard" and by_stage["guardrails"]["tokens"] > 0
    assert by_stage["embed_text"]["token_kind"] == "bge" and by_stage["embed_text"]["tokens"] > 0
    assert by_stage["replace"]["detail"]["children"] > 0


def test_a_rejected_attempt_is_recorded_and_a_summary_covers_only_ingested_ones(tables):
    from ingestion.errors import DocumentRejected

    storage = Storage()
    make_pipeline(storage, "lung ")("paper.pdf", intake())
    bad = make_pipeline(storage, "x", name="land-cover/bad.pdf")
    bad.extract = lambda p, i: extraction.ExtractionResult(source_name="land-cover/bad.pdf", blocks=[], pages=3,
                                                          page_chars=[0, 0, 0])
    with pytest.raises(DocumentRejected):
        bad("bad.pdf", intake("e" * 64, name="land-cover/bad.pdf"))

    with db.get_connection() as c:
        latest = ingestion_reports.recent(c, 5)
        assert [r["status"] for r in latest] == ["rejected", "ingested"] and latest[0]["reason_code"]
        summary = {r["stage"]: r for r in ingestion_reports.summary(c)}
    assert summary["extract"]["documents"] == 1  # the rejected attempt is not in the averages
    assert summary["guardrails"]["tokens"] > 0


def test_removing_a_document_deletes_its_rows_its_cache_entries_and_marks_the_manifest(tables):
    storage = Storage()
    make_pipeline(storage, "lung ", images=[{"page": 1, "filename": "a.png", "ext": "png", "bytes": b"i"}],
                  table_text="| a |")("paper.pdf", intake())
    other = "land-cover/other.pdf"
    make_pipeline(storage, "satellite ", name=other)("other.pdf", intake("f" * 64, name=other))
    with db.get_connection() as c:
        for hash_, text, chunks in (("h1", "q1", [{"source_pdf": NAME}]), ("h2", "q2", [{"source_pdf": other}]),
                                    ("h3", "q3", [{"source_pdf": NAME}, {"source_pdf": other}])):
            query_cache.write_cache(c, text, chunks, "answer")

    with db.get_connection() as c:
        plan = remove_document.plan_removal(c, NAME)
        assert plan.rows["text_chunks"] > 0 and plan.rows["images"] == 1 and plan.rows["doc_tables"] == 1
        assert plan.domain == "lung-cancer" and plan.cache_entries == 2
        remove_document.remove_document(c, NAME)

    assert all(count(tables, t, "source_pdf = %s", (NAME,)) == 0 for t in ("text_chunks", "text_parents", "doc_tables", "images"))
    assert count(tables, "text_chunks", "source_pdf = %s", (other,)) > 0           # the other document is untouched
    assert count(tables, "query_cache") == 1                                         # only the entry that cites just `other`
    with db.get_connection() as c:
        assert ingestion_manifest.get_active_hash(c, NAME) is None                   # marked deleted
        assert ingestion_manifest.get_active_hash(c, other) == "f" * 64
        assert c.execute(f"SELECT status FROM {tables}.ingestion_manifest WHERE source_pdf = %s", (NAME,)).fetchone()[0] == "deleted"
        assert [r[0] for r in c.execute(f"SELECT name FROM {tables}.domains ORDER BY name").fetchall()] == ["land-cover"]
        with pytest.raises(remove_document.DocumentNotFound):
            remove_document.remove_document(c, NAME)

    # the same file can be uploaded again afterwards
    report = make_pipeline(storage, "lung ")("paper.pdf", intake())
    assert report.domain_created and count(tables, "text_chunks", "source_pdf = %s", (NAME,)) > 0
