"""document_loader: writes ONE ingested document to Postgres, replacing any earlier version of it.

Everything happens in the caller's transaction (nothing is committed here), so a document is either fully
replaced or not touched: the old chunks, parents, tables and images of the same file are deleted, the new ones are
inserted, and, when this is a replacement, the cached answers that were built from the old version are removed
(the table belongs to the app, but only a DELETE on it is needed here). When a document is added or changed,
the WHOLE answer cache is cleared: a cached answer built before this document
existed may be improved by it, and clearing is always correct.
"""
from dataclasses import dataclass

import numpy as np

from shared.db import SCHEMA_NAME


@dataclass
class LoadCounts:
    parents: int = 0
    children: int = 0
    tables: int = 0
    images: int = 0
    cache_entries_removed: int = 0


def link_tables_to_parents(tables: list, parents: list) -> list:
    """The parent chunk each table belongs to (its id, or None), in the order of `tables`.

    A table is linked through its section heading: among the parents of that section, the one whose page range
    holds the table's page, else the first. A table whose heading matches no parent goes to the parent that holds
    its page. Linking by heading and page, not by page alone, keeps a table off the wrong section on a page that
    holds several."""
    linked = []
    for table in tables:
        page = table.get("page")
        same_section = [p for p in parents if p.get("section") == table.get("section_heading")]
        on_page = [p for p in same_section if _holds_page(p, page)]
        if on_page:
            chosen = on_page[0]
        elif same_section:
            chosen = same_section[0]
        else:
            chosen = next((p for p in parents if _holds_page(p, page)), None)
        linked.append(chosen["chunk_id"] if chosen else None)
    return linked


def _holds_page(parent: dict, page) -> bool:
    start, end = parent.get("page_start"), parent.get("page_end")
    return page is not None and start is not None and end is not None and start <= page <= end


def link_images_to_parents(images: list, parents: list) -> list:
    """The parent chunk each image belongs to (its id, or None), in the order of `images`: the first parent whose
    page range holds the image's page (an image has a page and no position inside it)."""
    linked = []
    for image in images:
        parent = next((p for p in parents if _holds_page(p, image.get("page"))), None)
        linked.append(parent["chunk_id"] if parent else None)
    return linked


def _vector(values):
    return np.asarray(values, dtype=np.float32)


def _insert_rows(conn, sql: str, rows: list) -> None:
    if rows:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)


DOCUMENT_TABLES = ("text_chunks", "text_parents", "doc_tables", "images")  # children before parents (the foreign key)


def count_document_rows(conn, source_pdf: str) -> dict:
    """How many rows each table holds for the document."""
    return {table: conn.execute(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{table} WHERE source_pdf = %s",
                                (source_pdf,)).fetchone()[0] for table in DOCUMENT_TABLES}


def delete_document_rows(conn, source_pdf: str) -> dict:
    """Deletes every row of the document from the data tables. Returns how many rows each table lost."""
    deleted = {}
    for table in DOCUMENT_TABLES:
        deleted[table] = conn.execute(f"DELETE FROM {SCHEMA_NAME}.{table} WHERE source_pdf = %s", (source_pdf,)).rowcount or 0
    return deleted


def replace_document(conn, *, source_pdf: str, stem: str, parents: list, child_vectors: list, tables: list,
                     table_vectors: list, images: list, domain: str, embedding_version: str,
                     clear_cache: bool = True) -> LoadCounts:
    """Replaces everything stored for `source_pdf` with the new version.

    parents      the chunk_documents.chunk_document parents, each holding its children
    child_vectors  one embedding per child, in the order the children appear across the parents
    tables       the guarded tables ({"text", "caption", "page", "section_heading"}); table_vectors is aligned with them
    images       [{"image_file", "page", "embedding"}]
    clear_cache  True (the default): the whole answer cache is cleared"""
    children = [child for parent in parents for child in parent["children"]]
    if len(children) != len(child_vectors) or len(tables) != len(table_vectors):
        raise ValueError("the embeddings do not line up with the chunks or the tables")

    counts = LoadCounts()
    s = SCHEMA_NAME

    delete_document_rows(conn, source_pdf)  # the old version

    if clear_cache:
        if conn.execute("SELECT to_regclass(%s)", (f"{s}.query_cache",)).fetchone()[0]:
            counts.cache_entries_removed = conn.execute(f"DELETE FROM {s}.query_cache").rowcount or 0

    _insert_rows(conn, f"""INSERT INTO {s}.text_parents (parent_id, source_pdf, section, page_start, page_end, domain, text)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                 [(p["chunk_id"], source_pdf, p["section"], p["page_start"], p["page_end"], domain, p["text"]) for p in parents])
    counts.parents = len(parents)

    _insert_rows(conn, f"""INSERT INTO {s}.text_chunks
                               (chunk_id, parent_id, source_pdf, section, page_start, page_end, domain, text,
                                embedding_version, embedding)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                 [(c["chunk_id"], c["parent_id"], source_pdf, c["section"], c["page_start"], c["page_end"], domain,
                   c["text"], embedding_version, _vector(v)) for c, v in zip(children, child_vectors)])
    counts.children = len(children)

    parent_ids = link_tables_to_parents(tables, parents)
    _insert_rows(conn, f"""INSERT INTO {s}.doc_tables
                               (table_id, source_pdf, page, section, parent_id, domain, caption, text, embedding_version,
                                embedding)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                 [(f"{stem}_t{n}", source_pdf, t.get("page"), t.get("section_heading"), parent_id, domain,
                   t.get("caption") or None, t["text"], embedding_version, _vector(v))
                  for n, (t, parent_id, v) in enumerate(zip(tables, parent_ids, table_vectors), 1)])
    counts.tables = len(tables)

    image_parents = link_images_to_parents(images, parents)
    _insert_rows(conn, f"""INSERT INTO {s}.images (image_file, source_pdf, page, parent_id, domain, embedding_version, embedding)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                 [(i["image_file"], source_pdf, i["page"], parent_id, domain, embedding_version, _vector(i["embedding"]))
                  for i, parent_id in zip(images, image_parents)])
    counts.images = len(images)
    return counts
