"""Orchestrator: runs the full ingestion pipeline, stage by stage, in order.

Stops on first failure (stages depend on each other's output - no point
continuing). Records every run + per-stage status/latency/tokens in
ingestion_runs/ingestion_run_stages (Postgres) - doubles as a human-readable
run history and as the state an unattended/scheduled run would resume from
automatically.

Does NOT call check_new_documents.py - that stays disabled/unwired (see its
own docstring). This always runs all stages against whatever is currently
in data/raw/, same as running each file by hand.

Uses a FRESH connection for every bookkeeping write, never one held open
across a stage - Neon (free tier) scales to zero after 5 min idle and will
silently close a long-held connection, which a running stage (e.g. Docling
extraction, several minutes) would exceed.
"""
import argparse
import time
import traceback

import usage_tracker
from db import get_connection, SCHEMA_NAME

from load_data import load_dataset
from extract_text import extract_text
from extract_images import extract_images
from prepare_documents import prepare_documents
from extract_entities import extract_entities
from classify_domain import classify_domain
from ingestion_guardrails import run_guardrails
from chunk_documents import chunk_documents
from embed_text import embed_text
from embed_images import embed_images
from load_vector_db import load_vector_db
from load_graph_db import load_graph_db

STAGES = [
    ("load_data", load_dataset),
    ("extract_text", extract_text),
    ("extract_images", extract_images),
    ("prepare_documents", prepare_documents),
    ("extract_entities", extract_entities),
    ("classify_domain", classify_domain),
    ("ingestion_guardrails", run_guardrails),
    ("chunk_documents", chunk_documents),
    ("embed_text", embed_text),
    ("embed_images", embed_images),
    ("load_vector_db", load_vector_db),
    ("load_graph_db", load_graph_db),
]
STAGE_NAMES = [name for name, _ in STAGES]


def _run_write(query, params=()):
    """Fresh connection per write - see module docstring."""
    conn = get_connection()
    result = conn.execute(query, params)
    row = result.fetchone() if result.description else None
    conn.commit()
    conn.close()
    return row


def _run_read(query, params=()):
    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def _find_resume_point():
    """Most recent run that didn't finish cleanly -> (run_id, first stage not
    yet succeeded). Returns (None, None) if the last run succeeded, or there
    is no prior run - i.e. start a fresh run from stage 1."""
    rows = _run_read(
        f"SELECT run_id FROM {SCHEMA_NAME}.ingestion_runs "
        f"WHERE status != 'succeeded' ORDER BY run_id DESC LIMIT 1"
    )
    if not rows:
        return None, None

    run_id = rows[0][0]
    stage_rows = _run_read(
        f"SELECT stage, status FROM {SCHEMA_NAME}.ingestion_run_stages "
        f"WHERE run_id = %s ORDER BY id",
        (run_id,),
    )
    succeeded = {stage for stage, status in stage_rows if status == "succeeded"}

    for name in STAGE_NAMES:
        if name not in succeeded:
            return run_id, name
    return None, None  # every stage already succeeded - treat as done, fresh run


def run_ingestion(start_from: str = None, trigger: str = "manual"):
    run_id = None
    if start_from is None:
        run_id, start_from = _find_resume_point()
        if start_from:
            print(f"Resuming unfinished run #{run_id} from stage '{start_from}'\n")

    start_index = STAGE_NAMES.index(start_from) if start_from else 0

    if run_id is None:
        run_id = _run_write(
            f"INSERT INTO {SCHEMA_NAME}.ingestion_runs (trigger) VALUES (%s) RETURNING run_id",
            (trigger,),
        )[0]

    print(f"Ingestion run #{run_id} - starting from stage '{STAGE_NAMES[start_index]}'\n")
    run_start = time.perf_counter()

    for name, func in STAGES[start_index:]:
        print(f"=== Stage: {name} ===")
        stage_id = _run_write(
            f"INSERT INTO {SCHEMA_NAME}.ingestion_run_stages (run_id, stage) "
            f"VALUES (%s, %s) RETURNING id",
            (run_id, name),
        )[0]

        before = usage_tracker.snapshot()
        stage_start = time.perf_counter()

        try:
            func()
        except Exception as e:
            elapsed = round(time.perf_counter() - stage_start, 2)
            tokens = usage_tracker.delta(before, usage_tracker.snapshot())
            _run_write(
                f"UPDATE {SCHEMA_NAME}.ingestion_run_stages SET status = 'failed', "
                f"finished_at = now(), latency_seconds = %s, prompt_tokens = %s, "
                f"output_tokens = %s, error_message = %s WHERE id = %s",
                (elapsed, tokens["prompt_tokens"], tokens["output_tokens"], str(e), stage_id),
            )
            _run_write(
                f"UPDATE {SCHEMA_NAME}.ingestion_runs SET status = 'failed', "
                f"finished_at = now() WHERE run_id = %s",
                (run_id,),
            )
            print(f"\nStage '{name}' FAILED after {elapsed}s: {e}")
            print(traceback.format_exc())
            print(f"\nRun #{run_id} stopped. Fix the issue and re-run - "
                  f"it will resume from '{name}' automatically.")
            return

        elapsed = round(time.perf_counter() - stage_start, 2)
        tokens = usage_tracker.delta(before, usage_tracker.snapshot())
        _run_write(
            f"UPDATE {SCHEMA_NAME}.ingestion_run_stages SET status = 'succeeded', "
            f"finished_at = now(), latency_seconds = %s, prompt_tokens = %s, output_tokens = %s "
            f"WHERE id = %s",
            (elapsed, tokens["prompt_tokens"], tokens["output_tokens"], stage_id),
        )
        print(f"=== Stage '{name}' done in {elapsed}s "
              f"({tokens['prompt_tokens']}+{tokens['output_tokens']} tokens) ===\n")

    total_elapsed = round(time.perf_counter() - run_start, 1)
    _run_write(
        f"UPDATE {SCHEMA_NAME}.ingestion_runs SET status = 'succeeded', "
        f"finished_at = now() WHERE run_id = %s",
        (run_id,),
    )

    print(f"\nIngestion run #{run_id} COMPLETE - "
          f"{len(STAGES) - start_index} stage(s) run, {total_elapsed}s total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline.")
    parser.add_argument("--start-from", default=None, choices=STAGE_NAMES,
                         help="Manual override: skip earlier stages, start from this one.")
    args = parser.parse_args()
    run_ingestion(start_from=args.start_from)
