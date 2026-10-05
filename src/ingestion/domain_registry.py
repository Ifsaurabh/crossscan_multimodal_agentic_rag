"""domain_registry: the domains of the documents.

The uploader decides the domain by choosing the folder: `incoming/<domain>/<file>`. A folder name that is not in the
registry yet is a NEW domain and is created here; no model is involved. The registry (see pipeline_schema.py) keeps:
  domains          name, document count, and the average embedding of its documents
  domain_members   one row per document: its domain and its FINGERPRINT (the average of its chunk embeddings)
A domain's average is the mean of its members' fingerprints, so replacing or removing a document only changes its
row. A domain left with no documents is removed (unless it was marked `seeded`, which keeps it).

Nothing here commits: the caller's transaction covers the whole ingestion of the document.
"""
from dataclasses import dataclass

import numpy as np

from shared.db import SCHEMA_NAME


@dataclass
class DomainAssignment:
    name: str
    created: bool   # the folder was a new domain, created for this document


def fingerprint(vectors) -> list:
    """The average of the chunk embeddings, scaled to unit length: the document's topic as one vector."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("a fingerprint needs at least one chunk embedding")
    mean = matrix.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return (mean / norm if norm else mean).tolist()


def _vector(values):
    return np.asarray(values, dtype=np.float32)


def list_domains(conn) -> list:
    rows = conn.execute(f"SELECT name, doc_count FROM {SCHEMA_NAME}.domains ORDER BY name").fetchall()
    return [{"name": name, "documents": count} for name, count in rows]


def refresh_domain(conn, name: str) -> None:
    """Recomputes a domain's document count and average from its members, and removes it if it is empty and
    was not seeded."""
    conn.execute(
        f"""UPDATE {SCHEMA_NAME}.domains d SET doc_count = m.n, centroid = m.c
            FROM (SELECT COUNT(*) AS n, AVG(fingerprint) AS c
                  FROM {SCHEMA_NAME}.domain_members WHERE domain = %s) m
            WHERE d.name = %s""",
        (name, name),
    )
    conn.execute(f"DELETE FROM {SCHEMA_NAME}.domains WHERE name = %s AND doc_count = 0 AND NOT seeded", (name,))


def release_document(conn, source_pdf: str):
    """Takes a document out of its domain (it is being removed). Returns the old domain, or None."""
    row = conn.execute(f"SELECT domain FROM {SCHEMA_NAME}.domain_members WHERE source_pdf = %s", (source_pdf,)).fetchone()
    if not row:
        return None
    conn.execute(f"DELETE FROM {SCHEMA_NAME}.domain_members WHERE source_pdf = %s", (source_pdf,))
    refresh_domain(conn, row[0])
    return row[0]


def assign_domain(conn, source_pdf: str, domain: str, fingerprint_vector) -> DomainAssignment:
    """Puts the document in `domain` (the upload folder), creating the domain if it is new, and stores the
    document's fingerprint. The caller commits."""
    created = conn.execute(
        f"INSERT INTO {SCHEMA_NAME}.domains (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING name",
        (domain,),
    ).fetchone() is not None

    conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.domain_members (source_pdf, domain, fingerprint) VALUES (%s, %s, %s)
            ON CONFLICT (source_pdf) DO UPDATE SET domain = EXCLUDED.domain, fingerprint = EXCLUDED.fingerprint,
                assigned_at = now()""",
        (source_pdf, domain, _vector(fingerprint_vector)),
    )
    refresh_domain(conn, domain)
    return DomainAssignment(name=domain, created=created)
