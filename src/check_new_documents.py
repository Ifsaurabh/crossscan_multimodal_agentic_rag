"""Nightly job: detect new/changed documents by content hash. Cheap and free
(no API calls, no embedding) - meant to run on a schedule.

FULLY DISABLED for now, double safety: the detection logic itself is
commented out below (not just the pipeline-trigger part), and this file
is not called from anywhere else in the project. Nothing here runs unless
someone deliberately uncomments it AND wires it in somewhere.
"""
from pathlib import Path

from db import get_connection
from ingestion_manifest import compute_content_hash, get_active_hash, is_up_to_date

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def check_new_documents():
    pass
    # conn = get_connection()
    #
    # new_docs = []
    # changed_docs = []
    # unchanged_count = 0
    #
    # for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
    #     source_pdf = pdf_path.name
    #     content_hash = compute_content_hash(pdf_path)
    #
    #     if is_up_to_date(conn, source_pdf, content_hash):
    #         unchanged_count += 1
    #         continue
    #
    #     if get_active_hash(conn, source_pdf) is None:
    #         new_docs.append(source_pdf)
    #     else:
    #         changed_docs.append(source_pdf)
    #
    # conn.close()
    #
    # print(f"Unchanged: {unchanged_count}")
    # print(f"New: {len(new_docs)} -> {new_docs}")
    # print(f"Changed: {len(changed_docs)} -> {changed_docs}")
    #
    # pending = new_docs + changed_docs
    # if not pending:
    #     print("\nNothing to do.")
    #     return pending
    #
    # print(f"\n{len(pending)} document(s) need ingestion. Pipeline run NOT triggered "
    #       f"(commented out below) - needs explicit approval since it spends Gemini quota.")
    #
    # # --- Pipeline trigger, disabled until an approval step exists ---
    # # for source_pdf in changed_docs:
    # #     archive_and_delete_old_content(source_pdf)   # data/raw_archive/, SQL + Cypher deletes
    # #
    # # for source_pdf in pending:
    # #     run_full_pipeline_for(source_pdf)             # extract_text.py through load_graph_db.py
    # #     mark_ingested(conn, source_pdf, compute_content_hash(RAW_DIR / source_pdf))
    #
    # return pending


if __name__ == "__main__":
    check_new_documents()
