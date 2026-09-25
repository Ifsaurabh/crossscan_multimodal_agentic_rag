"""ingestion_manifest: content hashes decide whether a document needs (re-)ingesting.
Not wired into the pipeline yet (check_new_documents.py is disabled), but the building
blocks are ready, so their behaviour is pinned here."""
import hashlib

import ingestion_manifest as im


class Cursor:
    def __init__(self, rows=(), rowcount=0):
        self.rows = list(rows)
        self.rowcount = rowcount

    def fetchone(self):
        return self.rows[0] if self.rows else None


class Conn:
    """Records every statement; answers the first response whose needle is in the SQL."""

    def __init__(self, responses=(), rowcount=0):
        self.responses = list(responses)
        self.rowcount = rowcount
        self.executed = []
        self.commits = 0
        self.timeline = []  # the order things happened in: statement kinds and COMMIT

    def commit(self):
        self.commits += 1
        self.timeline.append("COMMIT")

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self.timeline.append(sql.split()[0])
        for needle, rows in self.responses:
            if needle in sql:
                return Cursor(rows, self.rowcount)
        return Cursor([], self.rowcount)


# ---------- content hash ----------

def test_the_content_hash_is_the_sha256_of_the_file_bytes(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7 some content")

    assert im.compute_content_hash(pdf) == hashlib.sha256(b"%PDF-1.7 some content").hexdigest()


def test_the_same_bytes_give_the_same_hash_under_any_file_name(tmp_path):
    first, second = tmp_path / "a.pdf", tmp_path / "renamed.pdf"
    first.write_bytes(b"identical")
    second.write_bytes(b"identical")

    assert im.compute_content_hash(first) == im.compute_content_hash(second)


def test_one_changed_byte_changes_the_hash(tmp_path):
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    first.write_bytes(b"version 1")
    second.write_bytes(b"version 2")

    assert im.compute_content_hash(first) != im.compute_content_hash(second)


def test_an_empty_file_still_has_a_hash(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")

    assert im.compute_content_hash(empty) == hashlib.sha256(b"").hexdigest()


# ---------- reading the manifest ----------

def test_the_active_hash_is_returned_for_a_known_document():
    conn = Conn(responses=[("ingestion_manifest", [("abc123",)])])

    assert im.get_active_hash(conn, "paper.pdf") == "abc123"


def test_a_never_ingested_document_has_no_active_hash():
    assert im.get_active_hash(Conn(), "paper.pdf") is None


def test_only_the_active_row_of_this_document_is_looked_up():
    conn = Conn()

    im.get_active_hash(conn, "paper.pdf")

    sql, params = conn.executed[0]
    assert "status = 'active'" in sql and "source_pdf = %s" in sql
    assert params == ("paper.pdf",)


def test_up_to_date_means_the_active_hash_equals_the_files_hash():
    conn = Conn(responses=[("ingestion_manifest", [("abc123",)])])

    assert im.is_up_to_date(conn, "paper.pdf", "abc123") is True
    assert im.is_up_to_date(conn, "paper.pdf", "changed") is False


def test_a_document_never_ingested_is_not_up_to_date():
    assert im.is_up_to_date(Conn(), "paper.pdf", "abc123") is False


# ---------- recording an ingestion ----------

def test_marking_ingested_soft_deletes_the_old_version_then_records_the_new_one():
    conn = Conn()

    im.mark_ingested(conn, "paper.pdf", "newhash")

    (soft_delete, soft_params), (insert, insert_params) = conn.executed
    assert "UPDATE" in soft_delete and "status = 'deleted'" in soft_delete and "deleted_at = now()" in soft_delete
    assert soft_params == ("paper.pdf", "newhash")
    assert "INSERT INTO" in insert and insert_params == ("paper.pdf", "newhash")


def test_the_soft_delete_only_touches_a_different_active_version():
    conn = Conn()

    im.mark_ingested(conn, "paper.pdf", "samehash")

    soft_delete = conn.executed[0][0]
    assert "status = 'active'" in soft_delete and "content_hash != %s" in soft_delete  # the same hash is never deleted


def test_recording_the_same_version_twice_is_harmless():
    conn = Conn()

    im.mark_ingested(conn, "paper.pdf", "samehash")

    assert "ON CONFLICT (source_pdf, content_hash) DO NOTHING" in conn.executed[1][0]


def test_marking_ingested_commits_once_after_both_statements():
    """Without the commit the rows are rolled back when the connection closes and every
    document would look new again on the next nightly check."""
    conn = Conn()

    im.mark_ingested(conn, "paper.pdf", "newhash")

    assert conn.timeline == ["UPDATE", "INSERT", "COMMIT"]  # saved together, after both
    assert conn.commits == 1


def test_reading_the_manifest_never_commits():
    conn = Conn(responses=[("ingestion_manifest", [("abc123",)])])

    im.get_active_hash(conn, "paper.pdf")
    im.is_up_to_date(conn, "paper.pdf", "abc123")

    assert conn.commits == 0


def test_old_versions_are_kept_not_erased_so_a_rollback_stays_possible():
    conn = Conn()

    im.mark_ingested(conn, "paper.pdf", "newhash")

    assert not any("DELETE FROM" in sql for sql, _ in conn.executed)


# ---------- the 90-day rollback window ----------

def test_purging_removes_only_rows_soft_deleted_longer_ago_than_the_window():
    conn = Conn(rowcount=4)

    removed = im.purge_expired(conn)

    sql, params = conn.executed[0]
    assert removed == 4
    assert "DELETE FROM" in sql and "status = 'deleted'" in sql and "deleted_at <" in sql
    assert params == (90,)  # the default rollback window


def test_purging_commits_so_the_deletion_is_actually_saved():
    conn = Conn(rowcount=2)

    removed = im.purge_expired(conn)

    assert removed == 2                                   # the count is still returned
    assert conn.timeline == ["DELETE", "COMMIT"] and conn.commits == 1


def test_the_window_can_be_changed():
    conn = Conn()

    im.purge_expired(conn, days=30)

    assert conn.executed[0][1] == (30,)


def test_purging_with_nothing_expired_removes_zero():
    assert im.purge_expired(Conn(rowcount=0)) == 0
