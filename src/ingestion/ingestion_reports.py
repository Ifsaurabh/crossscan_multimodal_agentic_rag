"""ingestion_reports: the record of every attempt to ingest a document, with the time and the tokens of every stage.

Two append-only tables (nothing is ever updated or deleted here, so the history of every document is kept):
    ingestion_reports        one row per attempt: the document and its domain, a REPORT VERSION (1 for the first
                             attempt on that document, 2 for the next, ...), the content hash, the outcome
                             (`ingested`, `rejected` or `failed`) with the reason, the versions that produced the
                             data (chunking, embedding, pipeline), the start and finish, and the counts
    ingestion_report_stages  one row per stage of that attempt, in order: seconds, tokens (and what kind of token),
                             and the details of the stage

Tokens are only counted where a model reads text: Prompt Guard (guardrails), the chunker's tokenizer (chunk) and
the embedding model's tokenizer (embed_text). The worker makes no model calls, so there are no LLM tokens.

    PYTHONPATH=src python -m ingestion.ingestion_reports --setup            # create the two tables
    PYTHONPATH=src python -m ingestion.ingestion_reports --recent 10        # the last attempts
    PYTHONPATH=src python -m ingestion.ingestion_reports --summary          # time and tokens per stage
"""
import argparse
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from shared.db import SCHEMA_NAME, get_connection

INGESTED = "ingested"
REJECTED = "rejected"
FAILED = "failed"


def setup_sql(schema: str) -> str:
    """The SQL that creates the two tables (and their indexes) in `schema`. Only creates."""
    return f"""
CREATE TABLE IF NOT EXISTS {schema}.ingestion_reports (
    report_id BIGSERIAL PRIMARY KEY,
    source_pdf TEXT NOT NULL,
    report_version INTEGER NOT NULL,
    domain TEXT,
    content_hash TEXT,
    version_status TEXT,
    status TEXT NOT NULL CHECK (status IN ('ingested', 'rejected', 'failed')),
    reason_code TEXT,
    reason TEXT,
    chunking_version TEXT,
    embedding_version TEXT,
    pipeline_version TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    total_seconds NUMERIC,
    counts JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    UNIQUE (source_pdf, report_version)
);

CREATE INDEX IF NOT EXISTS ingestion_reports_started_idx
    ON {schema}.ingestion_reports (started_at DESC);

CREATE TABLE IF NOT EXISTS {schema}.ingestion_report_stages (
    id BIGSERIAL PRIMARY KEY,
    report_id BIGINT NOT NULL REFERENCES {schema}.ingestion_reports(report_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    stage TEXT NOT NULL,
    seconds NUMERIC NOT NULL,
    tokens BIGINT,
    token_kind TEXT,
    detail JSONB NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE INDEX IF NOT EXISTS ingestion_report_stages_report_idx
    ON {schema}.ingestion_report_stages (report_id, position);
"""


SETUP_SQL = setup_sql(SCHEMA_NAME)

COUNT_FIELDS = ("pages", "ocr_used", "parents", "children", "tables", "images", "images_skipped", "pii_redactions",
                "sections_dropped", "tables_dropped", "injection_flags", "cache_entries_removed", "domain_created", "image_bytes")


def save_report(conn, report, *, status: str, started_at: datetime, finished_at: datetime = None, reason_code: str = None,
                reason: str = None, chunking_version: str = None, embedding_version: str = None,
                pipeline_version: str = None) -> int:
    """Stores one attempt (an `IngestReport` from pipeline.py, possibly partial) and its stages. Returns the report id.
    The caller commits. The report version is the number of earlier attempts on the same document plus one."""
    finished_at = finished_at or datetime.now(timezone.utc)
    counts = {name: getattr(report, name) for name in COUNT_FIELDS if getattr(report, name, None) is not None}
    row = conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.ingestion_reports
                (source_pdf, report_version, domain, content_hash, version_status, status, reason_code, reason,
                 chunking_version, embedding_version, pipeline_version, started_at, finished_at, total_seconds, counts)
            VALUES (%s,
                    (SELECT COALESCE(MAX(report_version), 0) + 1 FROM {SCHEMA_NAME}.ingestion_reports WHERE source_pdf = %s),
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING report_id""",
        (report.source_pdf, report.source_pdf, report.domain, report.content_hash, report.version_status, status,
         reason_code, (reason or "")[:2000] or None, chunking_version, embedding_version, pipeline_version, started_at,
         finished_at, round(sum(s["seconds"] for s in report.stages), 2), Jsonb(counts)),
    ).fetchone()
    report_id = row[0]
    if report.stages:
        with conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {SCHEMA_NAME}.ingestion_report_stages (report_id, position, stage, seconds, tokens, token_kind, detail)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                [(report_id, position, s["stage"], s["seconds"], s.get("tokens"), s.get("token_kind"),
                  Jsonb(s.get("detail") or {})) for position, s in enumerate(report.stages, 1)],
            )
    return report_id


def recent(conn, limit: int = 10) -> list:
    rows = conn.execute(
        f"""SELECT report_id, source_pdf, report_version, domain, status, reason_code, total_seconds, started_at
            FROM {SCHEMA_NAME}.ingestion_reports ORDER BY report_id DESC LIMIT %s""", (limit,)).fetchall()
    keys = ("report_id", "source_pdf", "report_version", "domain", "status", "reason_code", "total_seconds", "started_at")
    return [dict(zip(keys, row)) for row in rows]


def stage_rows(conn, report_id: int) -> list:
    rows = conn.execute(
        f"""SELECT position, stage, seconds, tokens, token_kind, detail
            FROM {SCHEMA_NAME}.ingestion_report_stages WHERE report_id = %s ORDER BY position""", (report_id,)).fetchall()
    return [dict(zip(("position", "stage", "seconds", "tokens", "token_kind", "detail"), row)) for row in rows]


def summary(conn) -> list:
    """Per stage, over the attempts that were ingested: how many, the average, the slowest, and the tokens."""
    rows = conn.execute(
        f"""SELECT s.stage, COUNT(*), ROUND(AVG(s.seconds), 2), ROUND(MAX(s.seconds), 2), ROUND(SUM(s.seconds), 2),
                   SUM(s.tokens), MAX(s.token_kind)
            FROM {SCHEMA_NAME}.ingestion_report_stages s
            JOIN {SCHEMA_NAME}.ingestion_reports r USING (report_id)
            WHERE r.status = 'ingested'
            GROUP BY s.stage, s.position ORDER BY MIN(s.position)""").fetchall()
    keys = ("stage", "documents", "average_seconds", "slowest_seconds", "total_seconds", "tokens", "token_kind")
    return [dict(zip(keys, row)) for row in rows]


def setup_ingestion_reports():
    conn = get_connection()
    conn.execute(SETUP_SQL)
    conn.commit()
    conn.close()
    print(f"ingestion_reports and ingestion_report_stages ready in schema '{SCHEMA_NAME}'.")


def _print_table(rows: list, columns: list) -> None:
    if not rows:
        print("(nothing recorded yet)")
        return
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in columns]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in rows:
        print("  ".join(str(row[c] if row[c] is not None else "-").ljust(w) for c, w in zip(columns, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", action="store_true", help="create the two tables (nothing existing is changed)")
    parser.add_argument("--recent", type=int, metavar="N", help="the last N attempts")
    parser.add_argument("--summary", action="store_true", help="time and tokens per stage over the ingested documents")
    parser.add_argument("--report", type=int, metavar="ID", help="the stages of one attempt")
    args = parser.parse_args(argv)
    if args.setup:
        setup_ingestion_reports()
        return
    if not (args.recent or args.summary or args.report):
        parser.print_help()
        return
    conn = get_connection()
    try:
        if args.recent:
            _print_table(recent(conn, args.recent), ["report_id", "source_pdf", "report_version", "domain", "status",
                                                      "reason_code", "total_seconds", "started_at"])
        if args.summary:
            _print_table(summary(conn), ["stage", "documents", "average_seconds", "slowest_seconds", "total_seconds",
                                         "tokens", "token_kind"])
        if args.report:
            _print_table(stage_rows(conn, args.report), ["position", "stage", "seconds", "tokens", "token_kind", "detail"])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
