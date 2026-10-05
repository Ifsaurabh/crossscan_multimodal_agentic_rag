"""pipeline_schema: the three NEW tables the ingestion pipeline needs, next to the existing ones (nothing existing
is altered or dropped).

    domains          the registry of domains (name, one-line description, how many documents, their average
                     embedding); `seeded` marks a domain the owner defined, which is never removed automatically
    domain_members   one row per document: its domain and its fingerprint (the average embedding of its chunks).
                     A domain's average is the mean of its members' fingerprints, so replacing or removing a
                     document just changes its row
    doc_tables       the tables of the documents: the markdown text, the page, the section, the parent chunk it
                     belongs to, and the embedding of the text (so a table can be found by search)

Create them once (only CREATE ... IF NOT EXISTS):
    PYTHONPATH=src python -m ingestion.pipeline_schema --setup
"""
import argparse

from shared.db import SCHEMA_NAME, get_connection
from ingestion import worker_lease
from shared.setup_vector_db import TEXT_EMBEDDING_DIM


def setup_sql(schema: str) -> str:
    """The SQL that creates the three tables (and their indexes) in `schema`."""
    return f"""
CREATE TABLE IF NOT EXISTS {schema}.domains (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    seeded BOOLEAN NOT NULL DEFAULT FALSE,
    doc_count INTEGER NOT NULL DEFAULT 0,
    centroid vector({TEXT_EMBEDDING_DIM}),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {schema}.domain_members (
    source_pdf TEXT PRIMARY KEY,
    domain TEXT NOT NULL REFERENCES {schema}.domains(name) ON UPDATE CASCADE,
    fingerprint vector({TEXT_EMBEDDING_DIM}) NOT NULL,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS domain_members_domain_idx
    ON {schema}.domain_members (domain);

CREATE TABLE IF NOT EXISTS {schema}.doc_tables (
    table_id TEXT PRIMARY KEY,
    source_pdf TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    parent_id TEXT,
    domain TEXT NOT NULL,
    caption TEXT,
    text TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    embedding vector({TEXT_EMBEDDING_DIM}) NOT NULL
);

-- The table's caption ("Table 3: ..."), embedded together with its text. For a schema made before this column.
ALTER TABLE {schema}.doc_tables ADD COLUMN IF NOT EXISTS caption TEXT;

CREATE INDEX IF NOT EXISTS doc_tables_parent_idx
    ON {schema}.doc_tables (parent_id);

CREATE INDEX IF NOT EXISTS doc_tables_source_idx
    ON {schema}.doc_tables (source_pdf);

CREATE INDEX IF NOT EXISTS doc_tables_embedding_idx
    ON {schema}.doc_tables USING hnsw (embedding vector_cosine_ops);
""" + worker_lease.setup_sql(schema)


SETUP_SQL = setup_sql(SCHEMA_NAME)


def setup_pipeline_schema():
    conn = get_connection()
    conn.execute(SETUP_SQL)
    conn.commit()
    conn.close()
    print(f"domains, domain_members and doc_tables ready in schema '{SCHEMA_NAME}'.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", action="store_true", help="create the three tables (nothing existing is changed)")
    args = parser.parse_args(argv)
    if args.setup:
        setup_pipeline_schema()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
