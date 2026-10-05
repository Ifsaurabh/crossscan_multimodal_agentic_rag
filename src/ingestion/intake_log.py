"""intake_log: one row for every intake check.

The worker records what intake_check found for each uploaded file: accepted, skipped as unchanged, or
rejected, with the reason, the document label and the page counts and labels. It is an append-only
history, next to (not inside) the ingestion manifest, so the live manifest table is not touched. Two uses:
  - the REVIEW LIST: the files whose latest check was a rejection, and why (`list_rejected`);
  - the later pipeline steps can ask which pages of a document need OCR (`latest_check`).

Create the table once (it is a new table; nothing existing changes):
    PYTHONPATH=src python -m ingestion.intake_log --setup
Show the review list:
    PYTHONPATH=src python -m ingestion.intake_log --rejected
"""
import argparse

from psycopg.types.json import Jsonb

from shared.db import SCHEMA_NAME, connection, get_connection

COLUMNS = [
    "id", "source_pdf", "content_hash", "checked_at", "outcome", "reason_code", "reason", "version_status",
    "file_type", "detected_format", "size_bytes", "document_label", "pages", "text_pages", "image_only_pages",
    "blank_pages", "page_labels",
]
_COLUMN_LIST = ", ".join(COLUMNS)

def setup_sql(schema: str) -> str:
    """The SQL that creates the table (and its index) in `schema`. Only creates: nothing is altered or dropped."""
    return f"""
CREATE TABLE IF NOT EXISTS {schema}.intake_log (
    id BIGSERIAL PRIMARY KEY,
    source_pdf TEXT NOT NULL,
    content_hash TEXT,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    outcome TEXT NOT NULL CHECK (outcome IN ('accept', 'skip_unchanged', 'reject')),
    reason_code TEXT,
    reason TEXT,
    version_status TEXT,
    file_type TEXT,
    detected_format TEXT,
    size_bytes BIGINT,
    document_label TEXT,
    pages INTEGER,
    text_pages INTEGER,
    image_only_pages INTEGER,
    blank_pages INTEGER,
    page_labels JSONB
);

CREATE INDEX IF NOT EXISTS intake_log_source_idx
    ON {schema}.intake_log (source_pdf, checked_at DESC);
"""


SETUP_SQL = setup_sql(SCHEMA_NAME)


def setup_intake_log():
    conn = get_connection()
    conn.execute(SETUP_SQL)
    conn.commit()
    conn.close()
    print(f"intake_log table ready in schema '{SCHEMA_NAME}'.")


def record_check(conn, result) -> int:
    """Stores one intake result (an intake_check.IntakeResult) and returns the new row's id."""
    cur = conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.intake_log
                (source_pdf, content_hash, outcome, reason_code, reason, version_status, file_type, detected_format,
                 size_bytes, document_label, pages, text_pages, image_only_pages, blank_pages, page_labels)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id""",
        (
            result.file_name, result.content_hash, result.outcome, result.reason_code, result.reason,
            result.version_status, result.file_type, result.detected_format, result.size_bytes,
            result.document_label, result.pages, result.text_pages, result.image_only_pages, result.blank_pages,
            Jsonb(result.page_labels),
        ),
    )
    row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


def _as_dict(row) -> dict:
    return dict(zip(COLUMNS, row))


def latest_check(conn, source_pdf: str, content_hash: str = None):
    """The most recent check of a document (of one exact version when content_hash is given), or None."""
    sql = f"SELECT {_COLUMN_LIST} FROM {SCHEMA_NAME}.intake_log WHERE source_pdf = %s"
    params = [source_pdf]
    if content_hash is not None:
        sql += " AND content_hash = %s"
        params.append(content_hash)
    sql += " ORDER BY checked_at DESC, id DESC LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    return _as_dict(row) if row else None


def list_rejected(conn, limit: int = 100) -> list:
    """The review list: documents whose most recent check was a rejection, newest first. A document that
    was rejected and later accepted (after a fixed re-upload) is no longer on it."""
    rows = conn.execute(
        f"""SELECT {_COLUMN_LIST} FROM (
                SELECT DISTINCT ON (source_pdf) {_COLUMN_LIST}
                FROM {SCHEMA_NAME}.intake_log
                ORDER BY source_pdf, checked_at DESC, id DESC
            ) latest
            WHERE outcome = 'reject'
            ORDER BY checked_at DESC
            LIMIT %s""",
        (limit,),
    ).fetchall()
    return [_as_dict(row) for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", action="store_true", help="create the intake_log table (a new table; changes nothing existing)")
    parser.add_argument("--rejected", action="store_true", help="print the review list: files whose latest check was a rejection")
    args = parser.parse_args(argv)

    if args.setup:
        setup_intake_log()
    elif args.rejected:
        with connection() as conn:
            rows = list_rejected(conn)
        if not rows:
            print("Nothing to review: no file's latest check is a rejection.")
        for row in rows:
            print(f"{row['checked_at']:%Y-%m-%d %H:%M}  {row['source_pdf']}  [{row['reason_code']}]  {row['reason']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
