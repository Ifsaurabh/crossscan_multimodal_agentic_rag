"""pipeline: one document through every stage, with the models, the bucket and the database replaced by fakes."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ingestion import document_loader, domain_registry, extraction, ingestion_manifest
from ingestion import intake_check as ic
from ingestion import pipeline as pl
from ingestion.errors import DocumentRejected

LONG = "word " * 200


def blocks(pages=2):
    out = []
    for page in range(1, pages + 1):
        out.append({"label": "section_header", "level": 1, "page": page, "text": f"Section {page}"})
        out.append({"label": "text", "level": None, "page": page, "text": LONG})
    return out


def extracted(pages=2, chars=1000, **overrides):
    values = dict(source_name="paper.pdf", blocks=blocks(pages), pages=pages, page_chars=[chars] * pages,
                  tables_found=0, tables_failed=[], ocr_used=False)
    values.update(overrides)
    return extraction.ExtractionResult(**values)


def intake(**overrides):
    values = dict(file_name="lung-cancer/paper.pdf", outcome=ic.ACCEPT, content_hash="h" * 64, version_status="new",
                  file_type="pdf", pages=2, document_label="text")
    values.update(overrides)
    return ic.IntakeResult(**values)


class FakeStorage:
    def __init__(self):
        self.uploads, self.deleted = {}, []

    def upload_bytes(self, name, data, content_type, cache_control=None):
        self.uploads[name] = (data, content_type, cache_control)

    def delete_prefix(self, prefix, keep=()):
        self.deleted.append((prefix, sorted(keep)))


@pytest.fixture
def db(monkeypatch):
    """Replaces the three database stages and records what they were given."""
    calls = SimpleNamespace(assigned=None, loaded=None, manifest=None, order=[], saved=[], lines=[])

    def recorder(conn, report, **kw):
        calls.saved.append(SimpleNamespace(report=report, **kw))
        calls.order.append("report")

    @contextmanager
    def connect():
        yield "conn"

    def assign(conn, name, domain, fingerprint):
        calls.assigned = SimpleNamespace(name=name, domain=domain, fingerprint=fingerprint)
        calls.order.append("domain")
        return domain_registry.DomainAssignment(domain, True)

    def replace(conn, **k):
        calls.loaded = k
        calls.order.append("load")
        return document_loader.LoadCounts(parents=len(k["parents"]), children=len(k["child_vectors"]),
                                          tables=len(k["tables"]), images=len(k["images"]), cache_entries_removed=3)

    def mark(conn, name, content_hash):
        calls.manifest = (name, content_hash)
        calls.order.append("manifest")

    monkeypatch.setattr(pl.domain_registry, "assign_domain", assign)
    monkeypatch.setattr(pl.document_loader, "replace_document", replace)
    monkeypatch.setattr(pl.ingestion_manifest, "mark_ingested", mark)
    monkeypatch.setattr(pl, "guard_document", lambda doc, domain="unclassified": (doc, {
        "pii_redactions": 2, "sections_dropped": [], "tables_dropped": [], "injection_flags": []}))
    calls.connect = connect
    calls.recorder = recorder
    return calls


def build(db, storage=None, result=None, images=(), image_embedder=None, **overrides):
    storage = storage or FakeStorage()
    options = dict(
        connect=db.connect, text_embedder=lambda texts: [[1.0, 0.0] for _ in texts],
        text_counter=lambda texts: (len(texts) * 10, 1), image_embedder=image_embedder or (lambda data: [0.0, 1.0]),
        extract=lambda path, intake_result: result or extracted(), image_lister=lambda path: list(images),
        guard_token_counter=lambda text: len(text.split()), recorder=db.recorder, emit=db.lines.append)
    options.update(overrides)
    return pl.Pipeline(storage, **options), storage


def test_a_good_document_is_loaded_and_the_manifest_is_written_last(db):
    pipeline, storage = build(db, images=[{"page": 1, "filename": "page1_img1.png", "ext": "png", "bytes": b"img"}])

    report = pipeline("paper.pdf", intake())

    assert db.order == ["domain", "load", "manifest", "report"] and db.manifest == ("lung-cancer/paper.pdf", "h" * 64)
    assert report.domain == "lung-cancer" and report.domain_created and report.children >= 2 and report.images == 1
    assert report.pii_redactions == 2 and report.cache_entries_removed == 3
    assert set(report.stage_seconds) >= {"extract", "chunk", "embed_text", "replace"}
    assert db.loaded["domain"] == "lung-cancer" and db.loaded["clear_cache"] is True
    assert db.loaded["images"][0]["image_file"] == "lung-cancer/paper/page1_img1.png"
    assert storage.uploads["images/lung-cancer/paper/page1_img1.png"] == (b"img", "image/png", pl.IMAGE_CACHE_CONTROL)
    assert storage.deleted == [("images/lung-cancer/paper/", ["images/lung-cancer/paper/page1_img1.png"])]


def test_the_whole_answer_cache_is_cleared_for_a_new_document_and_for_a_changed_one(db):
    for status in ("new", "changed"):
        pipeline, _ = build(db)
        pipeline("paper.pdf", intake(version_status=status))
        assert db.loaded["clear_cache"] is True


def test_a_table_is_embedded_with_its_caption(db):
    texts = []
    blocks = [{"label": "section_header", "level": 1, "page": 1, "text": "Results"},
              {"label": "text", "level": None, "page": 1, "text": LONG},
              {"label": "table", "level": None, "page": 1, "text": "| a | b |", "caption": "Table 2: accuracy"}]
    pipeline, _ = build(db, result=extracted(pages=1, chars=1000, blocks=blocks, page_chars=[1000]),
                        text_embedder=lambda t: (texts.extend(t), [[1.0, 0.0] for _ in t])[1])
    pipeline("paper.pdf", intake(pages=1))
    assert texts[-1] == "Table 2: accuracy\n| a | b |"
    assert db.loaded["tables"][0]["caption"] == "Table 2: accuracy"


def test_the_domain_is_the_upload_folder_and_the_fingerprint_comes_from_the_chunk_embeddings(db):
    pipeline, _ = build(db)
    pipeline("paper.pdf", intake(file_name="Land-Cover/paper.pdf".lower()))
    assert (db.assigned.name, db.assigned.domain) == ("land-cover/paper.pdf", "land-cover")
    assert db.assigned.fingerprint == pytest.approx([1.0, 0.0])


def test_chunk_ids_and_pictures_are_unique_across_domains(db):
    pipeline, _ = build(db)
    pipeline("paper.pdf", intake(file_name="ai-security/paper.pdf"))
    assert db.loaded["stem"] == "ai-security/paper"
    assert db.loaded["parents"][0]["chunk_id"].startswith("ai-security/paper_p")
    # the text that is embedded names the file, not the domain: the domain is metadata only
    assert all(c["source_pdf"] == "paper.pdf" for p in db.loaded["parents"] for c in p["children"])


def test_a_name_without_a_domain_is_rejected(db):
    pipeline, _ = build(db)
    with pytest.raises(DocumentRejected) as rejected:
        pipeline("paper.pdf", intake(file_name="paper.pdf"))
    assert rejected.value.reason_code == pl.NO_DOMAIN and db.order == ["report"]


def test_a_poor_extraction_is_rejected_before_anything_is_stored(db):
    pipeline, storage = build(db, result=extracted(chars=10))
    with pytest.raises(DocumentRejected) as rejected:
        pipeline("paper.pdf", intake())
    assert rejected.value.reason_code and db.order == ["report"] and storage.uploads == {}


def test_a_document_with_only_references_is_rejected(db):
    only_refs = extracted(blocks=[{"label": "section_header", "level": 1, "page": 1, "text": "References"},
                                  {"label": "text", "level": None, "page": 1, "text": LONG}] * 1,
                          pages=1, page_chars=[1000])
    pipeline, _ = build(db, result=only_refs)
    with pytest.raises(DocumentRejected) as rejected:
        pipeline("paper.pdf", intake(pages=1))
    assert rejected.value.reason_code == pl.NO_TEXT and db.order == ["report"]


def test_a_document_where_the_guardrails_drop_everything_is_rejected(db, monkeypatch):
    monkeypatch.setattr(pl, "guard_document", lambda doc, domain="unclassified": ({**doc, "sections": []}, {
        "pii_redactions": 0, "sections_dropped": [{"heading": "x"}], "tables_dropped": [], "injection_flags": []}))
    pipeline, _ = build(db)
    with pytest.raises(DocumentRejected) as rejected:
        pipeline("paper.pdf", intake())
    assert rejected.value.reason_code == pl.UNSAFE_CONTENT


def test_a_picture_that_cannot_be_embedded_is_skipped_not_fatal(db):
    def bad(data):
        raise ValueError("not an image")

    pipeline, storage = build(db, images=[{"page": 1, "filename": "a.png", "ext": "png", "bytes": b"x"}],
                              image_embedder=bad)
    report = pipeline("paper.pdf", intake())
    assert report.images == 0 and report.images_skipped == 1 and storage.uploads == {}


def test_image_files_are_not_scanned_for_embedded_pictures(db):
    pipeline, storage = build(db, images=[{"page": 1, "filename": "a.png", "ext": "png", "bytes": b"x"}])
    pipeline("scan.png", intake(file_name="lung-cancer/scan.png", file_type="image"))
    assert storage.uploads == {} and storage.deleted == []


def test_a_database_error_propagates_so_the_message_is_retried(db, monkeypatch):
    def boom(conn, **k):
        raise RuntimeError("connection lost")

    monkeypatch.setattr(pl.document_loader, "replace_document", boom)
    pipeline, _ = build(db)
    with pytest.raises(RuntimeError):
        pipeline("paper.pdf", intake())
    assert db.manifest is None


def test_every_stage_is_timed_in_order_and_printed_live(db):
    pipeline, _ = build(db, images=[{"page": 1, "filename": "a.png", "ext": "png", "bytes": b"img"}])
    report = pipeline("paper.pdf", intake())

    assert [s["stage"] for s in report.stages] == ["extract", "quality", "prepare", "guardrails", "chunk", "embed_text",
                                                  "images", "domain", "replace", "manifest_commit"]
    assert all(s["seconds"] >= 0 for s in report.stages)
    assert len(db.lines) == len(report.stages) and "[lung-cancer/paper.pdf] extract" in db.lines[0]
    assert "tokens" in db.lines[3]  # the guardrails line carries its tokens


def test_tokens_are_counted_where_a_model_reads_text(db):
    pipeline, _ = build(db)
    report = pipeline("paper.pdf", intake())
    by_stage = {s["stage"]: s for s in report.stages}

    assert by_stage["guardrails"]["token_kind"] == "prompt_guard" and by_stage["guardrails"]["tokens"] > 0
    assert by_stage["chunk"]["token_kind"] == "tiktoken" and by_stage["chunk"]["tokens"] > 0
    assert by_stage["embed_text"]["token_kind"] == "bge" and by_stage["embed_text"]["detail"]["truncated"] == 1
    assert by_stage["extract"]["tokens"] is None and by_stage["load" if "load" in by_stage else "replace"]["tokens"] is None
    assert by_stage["replace"]["detail"]["children"] == report.children


def test_a_stage_without_a_counter_reports_no_tokens(db):
    pipeline, _ = build(db, text_counter=None, guard_token_counter=lambda text: None)
    by_stage = {s["stage"]: s for s in pipeline("paper.pdf", intake()).stages}
    assert by_stage["embed_text"]["tokens"] is None and by_stage["guardrails"]["tokens"] is None


def test_the_load_is_split_into_domain_replace_and_commit_timings(db):
    pipeline, _ = build(db)
    stages = [s["stage"] for s in pipeline("paper.pdf", intake()).stages]
    assert stages[-3:] == ["domain", "replace", "manifest_commit"]


def test_an_ingested_attempt_is_recorded_with_its_versions(db):
    pipeline, _ = build(db)
    pipeline("paper.pdf", intake())
    saved = db.saved[0]
    assert saved.status == "ingested" and saved.reason_code is None
    assert saved.embedding_version and saved.chunking_version and saved.pipeline_version == pl.PIPELINE_VERSION
    assert saved.report.stages and saved.finished_at >= saved.started_at


def test_a_rejected_attempt_is_recorded_with_the_stages_that_ran(db):
    pipeline, _ = build(db, result=extracted(chars=10))
    with pytest.raises(DocumentRejected):
        pipeline("paper.pdf", intake())
    saved = db.saved[0]
    assert saved.status == "rejected" and saved.reason_code
    assert [s["stage"] for s in saved.report.stages] == ["extract", "quality"]


def test_a_failed_attempt_is_recorded_and_the_error_still_reaches_the_worker(db, monkeypatch):
    def boom(conn, **k):
        raise RuntimeError("connection lost")

    monkeypatch.setattr(pl.document_loader, "replace_document", boom)
    pipeline, _ = build(db)
    with pytest.raises(RuntimeError):
        pipeline("paper.pdf", intake())
    saved = db.saved[0]
    assert saved.status == "failed" and saved.reason_code == "RuntimeError"
    assert saved.report.stages[-1]["stage"] == "replace" and "connection lost" in saved.report.stages[-1]["detail"]["error"]


def test_failing_to_save_the_report_never_changes_the_outcome(db):
    def broken(conn, report, **k):
        raise RuntimeError("reports table missing")

    pipeline, _ = build(db, recorder=broken)
    report = pipeline("paper.pdf", intake())   # the document is ingested all the same
    assert report.children > 0
    assert any("could not be saved" in line for line in db.lines)


def test_no_recorder_means_no_record(db):
    pipeline, _ = build(db, recorder=None)
    pipeline("paper.pdf", intake())
    assert db.saved == []
