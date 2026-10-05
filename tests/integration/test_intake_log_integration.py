"""Checks the intake_log SQL against a REAL Postgres (throwaway schema), because the unit tests use a stand-in
connection that would accept SQL a real database rejects (DISTINCT ON, JSONB, CHECK constraints, RETURNING).

Run:  python -m pytest tests/integration/test_intake_log_integration.py --run-integration
Nothing outside the throwaway `it_...` schema is touched (see conftest.py)."""
import psycopg
import pytest

from ingestion import intake_check as ic
from ingestion import intake_log as il

pytestmark = pytest.mark.integration


@pytest.fixture
def log(schema, conn, monkeypatch):
    """The intake_log table, created in the throwaway schema, with the module pointed at it."""
    assert schema.startswith("it_"), "the intake_log tests must only ever run in the throwaway schema"
    monkeypatch.setattr(il, "SCHEMA_NAME", schema)
    conn.execute(il.setup_sql(schema))
    conn.commit()
    conn.execute(f"TRUNCATE {schema}.intake_log")
    conn.commit()
    return conn


def accepted(name="paper.pdf", **overrides):
    values = dict(
        file_name=name, outcome=ic.ACCEPT, content_hash="a" * 64, size_bytes=5000, version_status="new",
        file_type="pdf", detected_format="PDF", pages=3, text_pages=2, image_only_pages=1, blank_pages=0,
        page_labels=["text", "text", "image_only"], document_label="mixed",
    )
    values.update(overrides)
    return ic.IntakeResult(**values)


def rejected(name="bad.pdf", **overrides):
    values = dict(
        file_name=name, outcome=ic.REJECT, reason_code=ic.CORRUPT, reason="The PDF could not be opened.",
        content_hash="b" * 64, size_bytes=900, file_type="pdf", detected_format="PDF",
    )
    values.update(overrides)
    return ic.IntakeResult(**values)


def test_the_setup_can_run_twice_without_error(log, schema):
    log.execute(il.setup_sql(schema))
    log.commit()


def test_a_check_round_trips_including_its_page_labels(log):
    new_id = il.record_check(log, accepted())

    found = il.latest_check(log, "paper.pdf")

    assert found["id"] == new_id and found["outcome"] == "accept" and found["document_label"] == "mixed"
    assert found["page_labels"] == ["text", "text", "image_only"]  # a real JSON list comes back
    assert (found["pages"], found["text_pages"], found["image_only_pages"], found["blank_pages"]) == (3, 2, 1, 0)
    assert found["checked_at"] is not None and found["size_bytes"] == 5000


def test_the_latest_check_is_the_newest_and_can_be_narrowed_to_one_version(log):
    il.record_check(log, accepted(content_hash="1" * 64, document_label="text"))
    il.record_check(log, accepted(content_hash="2" * 64, document_label="scanned"))

    assert il.latest_check(log, "paper.pdf")["content_hash"] == "2" * 64
    assert il.latest_check(log, "paper.pdf", content_hash="1" * 64)["document_label"] == "text"
    assert il.latest_check(log, "other.pdf") is None


def test_a_rejection_keeps_its_reason(log):
    il.record_check(log, rejected())

    found = il.latest_check(log, "bad.pdf")

    assert found["outcome"] == "reject" and found["reason_code"] == "corrupt" and found["document_label"] is None


def test_the_review_list_holds_files_whose_latest_check_was_a_rejection(log):
    il.record_check(log, rejected("first.pdf"))
    il.record_check(log, rejected("second.pdf", reason_code=ic.PASSWORD_PROTECTED))
    il.record_check(log, accepted("fine.pdf"))

    listed = il.list_rejected(log)

    assert [r["source_pdf"] for r in listed] == ["second.pdf", "first.pdf"]  # newest first
    assert listed[0]["reason_code"] == "password_protected"


def test_a_file_that_was_rejected_and_later_accepted_leaves_the_review_list(log):
    il.record_check(log, rejected("paper.pdf"))
    il.record_check(log, accepted("paper.pdf"))

    assert il.list_rejected(log) == []


def test_the_review_list_respects_its_limit(log):
    for n in range(5):
        il.record_check(log, rejected(f"bad{n}.pdf"))

    assert len(il.list_rejected(log, limit=2)) == 2


def test_an_unknown_outcome_is_refused_by_the_table(log):
    with pytest.raises(psycopg.errors.CheckViolation):
        il.record_check(log, accepted(outcome="maybe"))
    log.rollback()
