"""The pure parts of the ingestion pipeline: domain naming/fingerprints, table linking, the schema SQL, errors."""
from types import SimpleNamespace

import numpy as np
import pytest

from ingestion import domain_registry as dr
from ingestion import document_loader as dl
from ingestion import pipeline_schema
from ingestion.errors import DocumentRejected


def test_fingerprint_is_the_unit_length_mean():
    f = np.array(dr.fingerprint([[1, 0], [0, 1]]))
    assert pytest.approx(float(np.linalg.norm(f))) == 1.0
    assert pytest.approx(f[0]) == f[1]


def test_fingerprint_needs_vectors():
    with pytest.raises(ValueError):
        dr.fingerprint([])


class FakeConn:
    """Records the statements; `new_domain` says whether the INSERT of the domain created a row."""

    def __init__(self, new_domain=True):
        self.new_domain, self.statements = new_domain, []

    def execute(self, sql, params=()):
        self.statements.append((" ".join(sql.split()), params))
        rows = [("x",)] if (self.new_domain and "RETURNING name" in sql) else []
        return SimpleNamespace(fetchone=lambda: rows[0] if rows else None)


def test_a_new_folder_creates_the_domain_and_stores_the_fingerprint():
    conn = FakeConn(new_domain=True)

    result = dr.assign_domain(conn, "amnesia/p.pdf", "amnesia", [1.0, 0.0])

    assert (result.name, result.created) == ("amnesia", True)
    sql = [s for s, _ in conn.statements]
    assert sql[0].startswith("INSERT INTO") and "domains (name)" in sql[0] and conn.statements[0][1] == ("amnesia",)
    assert "domain_members" in sql[1] and conn.statements[1][1][:2] == ("amnesia/p.pdf", "amnesia")
    assert any(s.startswith("UPDATE") and "doc_count" in s for s in sql)  # the count and average are refreshed


def test_an_existing_folder_joins_the_domain_without_creating_it():
    assert dr.assign_domain(FakeConn(new_domain=False), "lung-cancer/p.pdf", "lung-cancer", [1.0]).created is False


def test_link_tables_by_section_then_page():
    parents = [
        {"chunk_id": "p1", "section": "Methods", "page_start": 1, "page_end": 2},
        {"chunk_id": "p2", "section": "Results", "page_start": 3, "page_end": 4},
        {"chunk_id": "p3", "section": "Results", "page_start": 5, "page_end": 6},
    ]
    tables = [
        {"section_heading": "Results", "page": 5},   # same section, page inside p3
        {"section_heading": "Results", "page": 9},   # same section, page nowhere: the first of that section
        {"section_heading": "Other", "page": 2},     # unknown heading: the parent that holds the page
        {"section_heading": "Other", "page": 99},    # nothing matches
        {"section_heading": "Methods", "page": None},
    ]
    assert dl.link_tables_to_parents(tables, parents) == ["p3", "p2", "p1", None, "p1"]


def test_replace_document_refuses_misaligned_embeddings():
    with pytest.raises(ValueError):
        dl.replace_document(object(), source_pdf="a.pdf", stem="a", parents=[{"children": [{}]}], child_vectors=[],
                            tables=[], table_vectors=[], images=[], domain="d", embedding_version="v",
                            clear_cache=False)


def test_schema_sql_is_per_schema_and_only_creates():
    sql = pipeline_schema.setup_sql("it_x")
    assert "it_x.domains" in sql and "it_x.domain_members" in sql and "it_x.doc_tables" in sql
    assert "DROP" not in sql.upper()
    # the only ALTERs add a column that a schema made earlier does not have yet
    assert all("ADD COLUMN IF NOT EXISTS" in line for line in sql.splitlines() if line.upper().startswith("ALTER"))


def test_document_rejected_carries_code_reason_details():
    e = DocumentRejected("too_empty", "empty", {"pages": 3})
    assert (e.reason_code, e.reason, e.details, str(e)) == ("too_empty", "empty", {"pages": 3}, "empty")


def test_an_image_is_linked_to_the_first_parent_that_holds_its_page():
    parents = [
        {"chunk_id": "p1", "page_start": 1, "page_end": 2},
        {"chunk_id": "p2", "page_start": 2, "page_end": 4},   # overlaps p1 on page 2
        {"chunk_id": "p3", "page_start": 5, "page_end": 6},
    ]
    images = [{"page": 1}, {"page": 2}, {"page": 4}, {"page": 6}, {"page": 9}, {"page": None}]

    assert dl.link_images_to_parents(images, parents) == ["p1", "p1", "p2", "p3", None, None]
