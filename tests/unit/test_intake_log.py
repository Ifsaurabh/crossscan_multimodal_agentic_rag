"""intake_log: the history of intake checks and the review list of rejected files. The database is a fake."""
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from psycopg.types.json import Jsonb

from ingestion import intake_check as ic
from ingestion import intake_log as il


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    """Records every statement with its parameters; answers with the next queued list of rows."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.executed = []
        self.commits = 0
        self.closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return FakeCursor(self.responses.pop(0) if self.responses else [])

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


def row(**overrides):
    base = {
        "id": 1, "source_pdf": "a.pdf", "content_hash": "h" * 64,
        "checked_at": datetime(2026, 10, 5, 9, 30, tzinfo=timezone.utc), "outcome": "reject",
        "reason_code": "blank", "reason": "Every page is blank.", "version_status": "new", "file_type": "pdf",
        "detected_format": "PDF", "size_bytes": 1234, "document_label": None, "pages": 2, "text_pages": 0,
        "image_only_pages": 0, "blank_pages": 2, "page_labels": ["blank", "blank"],
    }
    base.update(overrides)
    return tuple(base[name] for name in il.COLUMNS)


def accepted_result():
    return ic.IntakeResult(
        file_name="paper.pdf", outcome=ic.ACCEPT, content_hash="c" * 64, size_bytes=5000, version_status="new",
        file_type="pdf", detected_format="PDF", pages=3, text_pages=2, image_only_pages=1, blank_pages=0,
        page_labels=["text", "text", "image_only"], document_label="mixed",
    )


# ---------- the table ----------

def test_the_setup_only_creates_a_new_table_and_never_alters_or_drops_anything():
    sql = il.SETUP_SQL

    assert "CREATE TABLE IF NOT EXISTS" in sql and "intake_log" in sql
    assert "ALTER" not in sql.upper() and "DROP" not in sql.upper()
    assert "ingestion_manifest" not in sql  # the live manifest table is not touched


def test_every_column_the_code_reads_is_defined_in_the_table():
    for column in il.COLUMNS:
        assert column in il.SETUP_SQL, column


def test_the_table_limits_outcomes_to_the_three_the_check_can_give_and_stores_the_labels_as_json():
    assert "CHECK (outcome IN ('accept', 'skip_unchanged', 'reject'))" in il.SETUP_SQL
    assert "page_labels JSONB" in il.SETUP_SQL
    assert "CREATE INDEX IF NOT EXISTS" in il.SETUP_SQL


def test_setup_creates_the_table_commits_and_closes(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(il, "get_connection", lambda: conn)

    il.setup_intake_log()

    assert conn.executed == [(il.SETUP_SQL, None)] and conn.commits == 1 and conn.closed


# ---------- writing ----------

def test_a_check_is_stored_with_every_field_and_the_new_id_is_returned():
    conn = FakeConn([[(41,)]])

    new_id = il.record_check(conn, accepted_result())

    sql, params = conn.executed[0]
    assert "INSERT INTO" in sql and "intake_log" in sql and "RETURNING id" in sql
    assert new_id == 41 and conn.commits == 1
    assert params[:14] == ("paper.pdf", "c" * 64, "accept", None, None, "new", "pdf", "PDF", 5000, "mixed", 3, 2, 1, 0)
    assert isinstance(params[14], Jsonb) and params[14].obj == ["text", "text", "image_only"]


def test_the_number_of_values_matches_the_number_of_placeholders():
    conn = FakeConn([[(1,)]])

    il.record_check(conn, accepted_result())

    sql, params = conn.executed[0]
    assert sql.count("%s") == len(params) == 15


def test_a_rejection_is_stored_with_its_reason():
    conn = FakeConn([[(2,)]])
    result = ic.IntakeResult(file_name="locked.pdf", outcome=ic.REJECT, reason_code=ic.PASSWORD_PROTECTED,
                             reason="The PDF is password-protected and cannot be read.", content_hash="d" * 64, size_bytes=900)

    il.record_check(conn, result)

    params = conn.executed[0][1]
    assert params[0] == "locked.pdf" and params[2] == "reject" and params[3] == "password_protected"
    assert "password-protected" in params[4]


# ---------- reading ----------

def test_the_latest_check_of_a_document_comes_back_as_a_dict():
    conn = FakeConn([[row(outcome="accept", reason_code=None, document_label="text")]])

    found = il.latest_check(conn, "a.pdf")

    sql, params = conn.executed[0]
    assert "ORDER BY checked_at DESC" in sql and "LIMIT 1" in sql and "content_hash = " not in sql
    assert params == ["a.pdf"]
    assert found["outcome"] == "accept" and found["document_label"] == "text" and set(found) == set(il.COLUMNS)


def test_the_latest_check_of_one_exact_version_filters_on_its_hash():
    conn = FakeConn([[row()]])

    il.latest_check(conn, "a.pdf", content_hash="abc")

    sql, params = conn.executed[0]
    assert "AND content_hash = %s" in sql and params == ["a.pdf", "abc"]


def test_a_document_that_was_never_checked_has_no_latest_check():
    assert il.latest_check(FakeConn([[]]), "never.pdf") is None


def test_the_review_list_is_the_latest_check_per_document_when_that_was_a_rejection():
    conn = FakeConn([[row(source_pdf="a.pdf"), row(source_pdf="b.pdf", id=2, reason_code="corrupt")]])

    rejected = il.list_rejected(conn, limit=10)

    sql, params = conn.executed[0]
    assert "DISTINCT ON (source_pdf)" in sql and "outcome = 'reject'" in sql
    assert params == (10,)
    assert [r["source_pdf"] for r in rejected] == ["a.pdf", "b.pdf"] and rejected[1]["reason_code"] == "corrupt"


def test_an_empty_review_list_is_an_empty_list():
    assert il.list_rejected(FakeConn([[]])) == []


# ---------- the command line ----------

def test_setup_on_the_command_line_creates_the_table(monkeypatch):
    called = []
    monkeypatch.setattr(il, "setup_intake_log", lambda: called.append(True))

    il.main(["--setup"])

    assert called == [True]


def test_the_rejected_option_prints_one_line_per_file(monkeypatch, capsys):
    @contextmanager
    def fake_connection():
        yield "conn"

    monkeypatch.setattr(il, "connection", fake_connection)
    monkeypatch.setattr(il, "list_rejected", lambda conn: [il._as_dict(row(source_pdf="bad.pdf", reason="locked", reason_code="password_protected"))])

    il.main(["--rejected"])

    out = capsys.readouterr().out
    assert "bad.pdf" in out and "[password_protected]" in out and "locked" in out and "2026-10-05 09:30" in out


def test_an_empty_review_list_says_so(monkeypatch, capsys):
    @contextmanager
    def fake_connection():
        yield "conn"

    monkeypatch.setattr(il, "connection", fake_connection)
    monkeypatch.setattr(il, "list_rejected", lambda conn: [])

    il.main(["--rejected"])

    assert "Nothing to review" in capsys.readouterr().out


def test_with_no_options_the_help_is_shown_and_the_database_is_not_touched(monkeypatch, capsys):
    monkeypatch.setattr(il, "get_connection", lambda: pytest.fail("must not connect"))
    monkeypatch.setattr(il, "connection", lambda: pytest.fail("must not connect"))

    il.main([])

    assert "--setup" in capsys.readouterr().out
