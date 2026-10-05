"""remove_document: what it plans, what it deletes (in one transaction), and the command's questions."""
from contextlib import contextmanager

import pytest

from fake_db import FakeConn
from ingestion import doc_storage, remove_document as rd

NAME = "lung-cancer/paper.pdf"


def conn_for(found=True, cache=True):
    responses = [("text_chunks WHERE", [(26,)] if found else [(0,)]), ("text_parents WHERE", [(16,)] if found else [(0,)]),
                 ("doc_tables WHERE", [(6,)] if found else [(0,)]), ("images WHERE", [(18,)] if found else [(0,)]),
                 ("FROM rag_v2.domain_members", [("lung-cancer",)] if found else []),
                 ("FROM rag_v2.ingestion_manifest", [("a" * 64,)] if found else []),
                 ("to_regclass", [("rag_v2.query_cache",)] if cache else [(None,)]),
                 ("COUNT(*) FROM rag_v2.query_cache", [(3,)])]
    return FakeConn(responses=responses)


@pytest.fixture(autouse=True)
def schema(monkeypatch):
    for module in (rd, rd.document_loader, rd.domain_registry):
        monkeypatch.setattr(module, "SCHEMA_NAME", "rag_v2")


class FakeStorage:
    def __init__(self):
        self.deleted = []

    def delete_prefix(self, prefix, keep=()):
        self.deleted.append(prefix)
        return 7


def connector(conn):
    @contextmanager
    def connect():
        yield conn
    return connect


def test_the_plan_counts_everything_without_deleting():
    conn = conn_for()
    plan = rd.plan_removal(conn, NAME)

    assert plan.rows == {"text_chunks": 26, "text_parents": 16, "doc_tables": 6, "images": 18}
    assert plan.domain == "lung-cancer" and plan.cache_entries == 3 and plan.active_version == "a" * 64 and plan.found
    assert not [s for s, _ in conn.executed if s.lstrip().upper().startswith(("DELETE", "UPDATE"))]


def test_a_name_that_is_not_stored_is_not_found():
    assert rd.plan_removal(conn_for(found=False), NAME).found is False
    with pytest.raises(rd.DocumentNotFound):
        rd.remove_document(conn_for(found=False), NAME)


def test_removal_deletes_rows_children_first_then_the_cache_and_marks_the_manifest():
    conn = conn_for()
    rd.remove_document(conn, NAME)

    statements = [" ".join(s.split()) for s, _ in conn.executed if s.lstrip().upper().startswith(("DELETE", "UPDATE"))]
    deletes = [s for s in statements if s.startswith("DELETE FROM rag_v2.") and "source_pdf = %s" in s]
    assert [d.split()[2].split(".")[1] for d in deletes][:4] == ["text_chunks", "text_parents", "doc_tables", "images"]
    assert any("DELETE FROM rag_v2.query_cache WHERE sources @> ARRAY[%s]::text[]" in s for s in statements)
    assert any("UPDATE rag_v2.ingestion_manifest SET status = 'deleted'" in s for s in statements)
    assert any("DELETE FROM rag_v2.domain_members" in s for s in statements)  # taken out of its domain
    assert conn.commits == 0  # the caller's transaction commits


def test_the_cache_is_skipped_when_the_table_does_not_exist():
    conn = conn_for(cache=False)
    rd.remove_document(conn, NAME)
    assert not [s for s, _ in conn.executed if "query_cache" in s and "to_regclass" not in s]


def test_the_description_says_what_will_happen():
    text = rd.describe(rd.plan_removal(conn_for(), NAME))
    assert NAME in text and "26 text_chunks" in text and "lung-cancer" in text and "3 that cite it" in text
    assert "images/lung-cancer/paper/" in text and "PDF in processed/ is not touched" in text


def test_a_dry_run_deletes_nothing(capsys):
    conn, storage = conn_for(), FakeStorage()
    assert rd.main([NAME, "--dry-run"], connect=connector(conn), storage=storage) == 0
    assert "Dry run" in capsys.readouterr().out and storage.deleted == []
    assert not [s for s, _ in conn.executed if s.lstrip().upper().startswith(("DELETE", "UPDATE"))]


def test_without_a_yes_the_command_asks_and_stops_on_anything_but_yes(capsys):
    conn, storage = conn_for(), FakeStorage()
    assert rd.main([NAME], connect=connector(conn), storage=storage, ask=lambda q: "no") == 1
    assert "Cancelled" in capsys.readouterr().out and storage.deleted == []


def test_confirming_deletes_the_rows_then_the_pictures():
    conn, storage = conn_for(), FakeStorage()
    assert rd.main([NAME], connect=connector(conn), storage=storage, ask=lambda q: "YES ") == 0
    assert any("UPDATE rag_v2.ingestion_manifest" in s for s, _ in conn.executed)
    assert storage.deleted == [doc_storage.image_prefix(NAME)] == ["images/lung-cancer/paper/"]


def test_yes_skips_the_question():
    conn, storage = conn_for(), FakeStorage()
    assert rd.main([NAME, "--yes"], connect=connector(conn), storage=storage, ask=lambda q: pytest.fail("no question")) == 0


def test_an_unknown_name_says_so_and_changes_nothing(capsys):
    storage = FakeStorage()
    assert rd.main(["x/nothing.pdf", "--yes"], connect=connector(conn_for(found=False)), storage=storage) == 1
    assert "Nothing is stored" in capsys.readouterr().out and storage.deleted == []


def test_the_picture_folder_of_a_document():
    assert doc_storage.image_prefix("land-cover/My Paper v2.pdf") == "images/land-cover/My Paper v2/"
