"""cutover: the data steps of switching the app from one schema to another (rag_new -> rag_v2).

    PYTHONPATH=src python -m retrieval.cutover --source rag_new --target rag_v2 --verify        # what is where; changes nothing
    PYTHONPATH=src python -m retrieval.cutover --source rag_new --target rag_v2 --copy --dry-run # what would be copied
    PYTHONPATH=src python -m retrieval.cutover --source rag_new --target rag_v2 --copy           # copy the app data
    PYTHONPATH=src python -m retrieval.cutover --target rag_v2 --flush-cache                     # the one-time cache flush

COPY copies the users, sessions, chats, usage, memories, feedback and online scores from the source to the target, keeping their
ids, and never deletes or changes anything in the source. It refuses to copy into a target that already holds app data (unless
--allow-nonempty, which skips rows that already exist). The documents, chunks, images, tables and the manifest are NOT copied:
they were ingested into the target by the worker. The answer cache is not copied either: flush it (--flush-cache) when the app is
switched, because its answers were made from the old data.

The switch itself is then one setting: SCHEMA_NAME=<target> on the Cloud Run service, in the same deploy as the new code.
Stopping the old app first (or doing the copy at a quiet moment) means no chat is written to the old schema after the copy.
"""
import argparse
import re
import sys

APP_TABLES = (  # parents before children (the foreign keys)
    "users", "auth_sessions", "chat_sessions", "chat_messages", "usage_daily", "user_memories", "answer_feedback",
    "online_eval_scores",
)
DATA_TABLES = ("text_parents", "text_chunks", "doc_tables", "images", "ingestion_manifest", "domains", "domain_members")
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class CutoverError(Exception):
    """Something that makes the copy unsafe (a bad schema name, a non-empty target)."""


def check_schema_name(name: str) -> str:
    if not _IDENTIFIER.match(name or ""):
        raise CutoverError(f"{name!r} is not a plain schema name")
    return name


def _columns(conn, schema: str, table: str) -> list:
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (schema, table)).fetchall()
    return [r[0] for r in rows]


def _count(conn, schema: str, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {schema}.{table}").fetchone()[0]


def _exists(conn, schema: str, table: str) -> bool:
    return bool(conn.execute("SELECT to_regclass(%s)", (f"{schema}.{table}",)).fetchone()[0])


def plan_copy(conn, source: str, target: str) -> list:
    """[(table, rows in the source, rows in the target)] for every app table both schemas have."""
    check_schema_name(source), check_schema_name(target)
    if source == target:
        raise CutoverError("the source and the target are the same schema")
    plan = []
    for table in APP_TABLES:
        if _exists(conn, source, table) and _exists(conn, target, table):
            plan.append((table, _count(conn, source, table), _count(conn, target, table)))
    return plan


def copy_app_data(conn, source: str, target: str, allow_nonempty: bool = False) -> dict:
    """Copies the app tables, keeping ids. Returns {table: rows copied}. The caller commits (one transaction: all or nothing)."""
    plan = plan_copy(conn, source, target)
    occupied = [table for table, _, in_target in plan if in_target]
    if occupied and not allow_nonempty:
        raise CutoverError(f"the target already holds rows in {', '.join(occupied)}; nothing was copied "
                           "(use --allow-nonempty to add only the rows that are missing)")
    copied = {}
    for table, _, _ in plan:
        shared = [c for c in _columns(conn, source, table) if c in set(_columns(conn, target, table))]
        column_list = ", ".join(shared)
        copied[table] = conn.execute(
            f"INSERT INTO {target}.{table} ({column_list}) SELECT {column_list} FROM {source}.{table} ON CONFLICT DO NOTHING"
        ).rowcount or 0
    fix_sequences(conn, target)
    return copied


def fix_sequences(conn, schema: str) -> None:
    """After copying rows with their ids, the counters of the id columns must start after the largest id, or the next new
    row would reuse an id."""
    rows = conn.execute(
        """SELECT table_name, column_name FROM information_schema.columns
           WHERE table_schema = %s AND column_default LIKE 'nextval(%%'""", (schema,)).fetchall()
    for table, column in rows:
        if table not in APP_TABLES:
            continue
        conn.execute(
            f"""SELECT setval(pg_get_serial_sequence(%s, %s), COALESCE((SELECT MAX({column}) FROM {schema}.{table}), 1),
                              (SELECT MAX({column}) IS NOT NULL FROM {schema}.{table}))""",
            (f"{schema}.{table}", column))


def flush_cache(conn, schema: str) -> int:
    """The one-time flush of the answer cache. Returns how many entries were removed."""
    check_schema_name(schema)
    if not _exists(conn, schema, "query_cache"):
        return 0
    return conn.execute(f"DELETE FROM {schema}.query_cache").rowcount or 0


def verify(conn, source: str, target: str) -> dict:
    """What is where, and whether the target is ready: the data was ingested, the new columns are filled, the app data is there."""
    check_schema_name(source), check_schema_name(target)
    report = {"app_tables": {}, "data_tables": {}, "problems": []}
    for table in APP_TABLES:
        report["app_tables"][table] = {
            "source": _count(conn, source, table) if _exists(conn, source, table) else None,
            "target": _count(conn, target, table) if _exists(conn, target, table) else None}
    for table in DATA_TABLES:
        report["data_tables"][table] = _count(conn, target, table) if _exists(conn, target, table) else None

    def problem(condition, text):
        if condition:
            report["problems"].append(text)

    data = report["data_tables"]
    problem(not data["text_chunks"], "the target has no text chunks: nothing was ingested into it yet")
    problem(data["ingestion_manifest"] is not None and data["ingestion_manifest"] == 0, "the target manifest is empty")
    if _exists(conn, target, "images"):
        missing = conn.execute(f"SELECT COUNT(*) FROM {target}.images WHERE parent_id IS NULL").fetchone()[0]
        problem(missing, f"{missing} image(s) have no parent chunk")
    if _exists(conn, target, "doc_tables"):
        missing = conn.execute(f"SELECT COUNT(*) FROM {target}.doc_tables WHERE parent_id IS NULL").fetchone()[0]
        problem(missing, f"{missing} table(s) have no parent chunk")
    for table, counts in report["app_tables"].items():
        if counts["source"] and not counts["target"]:
            problem(True, f"{table}: {counts['source']} row(s) in the source, none in the target (run --copy)")
        elif counts["source"] and counts["target"] < counts["source"]:
            problem(True, f"{table}: the target has fewer rows ({counts['target']}) than the source ({counts['source']})")
    return report


def main(argv=None, connect=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="rag_new")
    parser.add_argument("--target", default="rag_v2")
    parser.add_argument("--verify", action="store_true", help="show what is where and what is missing; changes nothing")
    parser.add_argument("--copy", action="store_true", help="copy the app data from the source to the target")
    parser.add_argument("--dry-run", action="store_true", help="with --copy: only show what would be copied")
    parser.add_argument("--allow-nonempty", action="store_true", help="with --copy: add the missing rows to a target that has some")
    parser.add_argument("--flush-cache", action="store_true", help="empty the target's answer cache (the one-time flush)")
    args = parser.parse_args(argv)
    if not (args.verify or args.copy or args.flush_cache):
        parser.print_help()
        return 0

    if connect is None:
        from shared.db import connection as connect

    with connect() as conn:
        if args.verify:
            report = verify(conn, args.source, args.target)
            print(f"App data ({args.source} -> {args.target}):")
            for table, counts in report["app_tables"].items():
                print(f"  {table:<18} source {counts['source']}  target {counts['target']}")
            print(f"Ingested data in {args.target}:")
            for table, count in report["data_tables"].items():
                print(f"  {table:<18} {count}")
            print("Ready." if not report["problems"] else "Not ready:\n" + "\n".join(f"  - {p}" for p in report["problems"]))
        if args.copy:
            plan = plan_copy(conn, args.source, args.target)
            for table, in_source, in_target in plan:
                print(f"  {table:<18} {in_source} row(s) in the source, {in_target} in the target")
            if args.dry_run:
                print("Dry run: nothing was copied.")
            else:
                try:
                    copied = copy_app_data(conn, args.source, args.target, args.allow_nonempty)
                except CutoverError as e:
                    print(f"Not copied: {e}")
                    return 1
                print("Copied: " + ", ".join(f"{t} {n}" for t, n in copied.items()))
        if args.flush_cache:
            print(f"Removed {flush_cache(conn, args.target)} cache entr(ies) from {args.target}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
