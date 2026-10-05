"""remove_document: deletes one document from the system.

    PYTHONPATH=src python -m ingestion.remove_document lung-cancer/paper.pdf              # shows what it will delete, asks
    PYTHONPATH=src python -m ingestion.remove_document lung-cancer/paper.pdf --dry-run    # only shows
    PYTHONPATH=src python -m ingestion.remove_document lung-cancer/paper.pdf --yes        # no question

In ONE database transaction it deletes the document's chunks, parents, tables and images, takes it out of its domain (a
domain left empty disappears, unless it was seeded), deletes the answer-cache entries that cite it, and marks its manifest
row `deleted`. After the commit it deletes the document's picture files from the bucket. The PDF itself (under
`processed/`) is NOT touched. Deleting a file in the bucket never triggers this: the worker's own moves also produce delete
events, so they could not be told apart.

Which database it works on follows SCHEMA_NAME, like every other command.
"""
import argparse
import sys
from dataclasses import dataclass, field

from ingestion import doc_storage, document_loader, domain_registry
from shared.db import SCHEMA_NAME


class DocumentNotFound(Exception):
    """Nothing is stored under that name."""


@dataclass
class RemovalPlan:
    name: str
    rows: dict = field(default_factory=dict)      # table -> rows
    domain: str = None
    cache_entries: int = 0
    active_version: str = None                    # the content hash of the active manifest row, if any

    @property
    def found(self) -> bool:
        return bool(any(self.rows.values()) or self.domain or self.active_version)


def _cache_exists(conn) -> bool:
    return bool(conn.execute("SELECT to_regclass(%s)", (f"{SCHEMA_NAME}.query_cache",)).fetchone()[0])


def plan_removal(conn, name: str) -> RemovalPlan:
    """What removing the document would delete. Changes nothing."""
    plan = RemovalPlan(name=name, rows=document_loader.count_document_rows(conn, name))
    row = conn.execute(f"SELECT domain FROM {SCHEMA_NAME}.domain_members WHERE source_pdf = %s", (name,)).fetchone()
    plan.domain = row[0] if row else None
    row = conn.execute(f"SELECT content_hash FROM {SCHEMA_NAME}.ingestion_manifest WHERE source_pdf = %s AND status = 'active'",
                       (name,)).fetchone()
    plan.active_version = row[0] if row else None
    if _cache_exists(conn):
        plan.cache_entries = conn.execute(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.query_cache WHERE sources @> ARRAY[%s]::text[]",
                                          (name,)).fetchone()[0]
    return plan


def remove_document(conn, name: str) -> RemovalPlan:
    """Deletes the document inside the caller's transaction (the caller commits). Returns what was deleted."""
    plan = plan_removal(conn, name)
    if not plan.found:
        raise DocumentNotFound(f"nothing is stored under {name!r}")
    document_loader.delete_document_rows(conn, name)
    domain_registry.release_document(conn, name)
    if _cache_exists(conn):
        conn.execute(f"DELETE FROM {SCHEMA_NAME}.query_cache WHERE sources @> ARRAY[%s]::text[]", (name,))
    conn.execute(f"UPDATE {SCHEMA_NAME}.ingestion_manifest SET status = 'deleted', deleted_at = now() "
                 f"WHERE source_pdf = %s AND status = 'active'", (name,))
    return plan


def describe(plan: RemovalPlan) -> str:
    rows = ", ".join(f"{n} {table}" for table, n in plan.rows.items())
    return "\n".join([
        f"Document: {plan.name}",
        f"  rows to delete:   {rows}",
        f"  domain:           {plan.domain or '(none)'} (left without documents it disappears)",
        f"  cache entries:    {plan.cache_entries} that cite it",
        f"  manifest:         {'active version ' + plan.active_version[:12] + ', will be marked deleted' if plan.active_version else '(no active version)'}",
        f"  picture files:    everything under {doc_storage.image_prefix(plan.name)} (deleted after the commit)",
        "  the PDF in processed/ is not touched",
    ])


def main(argv=None, connect=None, storage=None, ask=input):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="the document's name: <domain>/<file>, as shown in the sources of an answer")
    parser.add_argument("--dry-run", action="store_true", help="only show what would be deleted")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args(argv)

    if connect is None:
        from shared.db import connection as connect
    name = args.name.strip()

    with connect() as conn:
        plan = plan_removal(conn, name)
    if not plan.found:
        print(f"Nothing is stored under {name!r}. Use the name shown in the sources: <domain>/<file>.")
        return 1
    print(describe(plan))
    if args.dry_run:
        print("Dry run: nothing was deleted.")
        return 0
    if not args.yes and ask("Delete this document? Type 'yes' to confirm: ").strip().lower() != "yes":
        print("Cancelled: nothing was deleted.")
        return 1

    with connect() as conn:
        remove_document(conn, name)  # the pool commits when the block ends without an error
    print("Deleted from the database.")

    storage = storage or doc_storage.DocumentStorage()
    removed = storage.delete_prefix(doc_storage.image_prefix(name))
    print(f"Deleted {removed} picture file(s) from the bucket.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
