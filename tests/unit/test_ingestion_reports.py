"""ingestion_reports: the SQL it builds and how it prints. (The real tables are covered by the integration tests.)"""
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

from ingestion import ingestion_reports as ir

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


class FakeConn:
    def __init__(self):
        self.executed, self.many = [], []

    def execute(self, sql, params=()):
        self.executed.append((" ".join(sql.split()), params))
        return SimpleNamespace(fetchone=lambda: (41,))

    @contextmanager
    def cursor(self):
        conn = self
        yield SimpleNamespace(executemany=lambda sql, rows: conn.many.append((" ".join(sql.split()), list(rows))))


def report(stages):
    return SimpleNamespace(source_pdf="lung-cancer/a.pdf", domain="lung-cancer", content_hash="h", version_status="new",
                           stages=stages, pages=2, children=5, domain_created=True, ocr_used=False, parents=None)


def test_setup_only_creates_two_tables_in_the_given_schema():
    sql = ir.setup_sql("it_x")
    assert "it_x.ingestion_reports" in sql and "it_x.ingestion_report_stages" in sql
    assert "DROP" not in sql.upper() and "ALTER" not in sql.upper() and sql.count("CREATE TABLE IF NOT EXISTS") == 2


def test_saving_gives_the_next_report_version_and_stores_each_stage_in_order():
    conn = FakeConn()
    stages = [{"stage": "extract", "seconds": 1.5, "tokens": None, "token_kind": None, "detail": {"pages": 2}},
              {"stage": "guardrails", "seconds": 2.0, "tokens": 900, "token_kind": "prompt_guard", "detail": {}}]

    report_id = ir.save_report(conn, report(stages), status="ingested", started_at=NOW, finished_at=NOW)

    sql, params = conn.executed[0]
    assert report_id == 41 and "MAX(report_version), 0) + 1" in sql and params[0] == params[1] == "lung-cancer/a.pdf"
    assert NOW in params  # the start time is stored
    assert 3.5 in params  # the total seconds is the sum of the stages
    rows = conn.many[0][1]
    assert [(r[1], r[2], r[3], r[4]) for r in rows] == [(1, "extract", 1.5, None), (2, "guardrails", 2.0, 900)]


def test_a_report_without_stages_still_saves_the_attempt():
    conn = FakeConn()
    ir.save_report(conn, report([]), status="rejected", started_at=NOW, reason_code="no_domain", reason="x" * 5000)
    assert conn.many == [] and len(conn.executed[0][1][7]) == 2000  # a long reason is cut


def test_unset_counts_are_left_out_of_the_counts():
    conn = FakeConn()
    ir.save_report(conn, report([]), status="ingested", started_at=NOW)
    counts = conn.executed[0][1][-1].obj
    assert counts["pages"] == 2 and counts["children"] == 5 and "parents" not in counts


def test_an_empty_table_prints_a_clear_line(capsys):
    ir._print_table([], ["a"])
    assert "nothing recorded" in capsys.readouterr().out
