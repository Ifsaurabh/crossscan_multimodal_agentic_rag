import hashlib
from pathlib import Path

from db import get_connection, SCHEMA_NAME


def compute_content_hash(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def get_active_hash(conn, source_pdf: str):
    """Returns the current active content hash for a document, or None if
    it has never been ingested."""
    row = conn.execute(
        f"SELECT content_hash FROM {SCHEMA_NAME}.ingestion_manifest "
        f"WHERE source_pdf = %s AND status = 'active'",
        (source_pdf,),
    ).fetchone()
    return row[0] if row else None


def is_up_to_date(conn, source_pdf: str, content_hash: str) -> bool:
    return get_active_hash(conn, source_pdf) == content_hash


def mark_ingested(conn, source_pdf: str, content_hash: str):
    """Soft-deletes any existing active row for this document (a prior
    version, if one exists) and records the new one as active. A no-op
    soft-delete if the hash is unchanged - MERGE-like, safe to call
    even when nothing actually changed. Commits once at the end, so both
    statements are saved together or not at all (without the commit the
    rows would be rolled back when the connection closes, and every
    document would look new again on the next check)."""
    conn.execute(
        f"UPDATE {SCHEMA_NAME}.ingestion_manifest SET status = 'deleted', deleted_at = now() "
        f"WHERE source_pdf = %s AND status = 'active' AND content_hash != %s",
        (source_pdf, content_hash),
    )
    conn.execute(
        f"INSERT INTO {SCHEMA_NAME}.ingestion_manifest (source_pdf, content_hash) "
        f"VALUES (%s, %s) ON CONFLICT (source_pdf, content_hash) DO NOTHING",
        (source_pdf, content_hash),
    )
    conn.commit()


def purge_expired(conn, days: int = 90) -> int:
    """Hard-deletes manifest rows soft-deleted more than `days` ago -
    the rollback window. Only removes the manifest record itself; does
    not touch Postgres/Neo4j content (a separate concern)."""
    cur = conn.execute(
        f"DELETE FROM {SCHEMA_NAME}.ingestion_manifest "
        f"WHERE status = 'deleted' AND deleted_at < now() - make_interval(days => %s)",
        (days,),
    )
    removed = cur.rowcount
    conn.commit()
    return removed
