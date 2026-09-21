import json
import os

from deepeval.synthesizer import Synthesizer
from google.genai.errors import ClientError

from db import get_connection
from deepeval_gemini_model import GeminiDeepEvalModel

GOLDEN_SET_PATH = os.path.join("data", "golden_set.jsonl")
MAX_GOLDENS_PER_CONTEXT = 1


def load_representative_chunks():
    """One parent chunk per source paper - the longest one, as the most
    content-rich representative - so every paper in the 12-paper corpus
    contributes to the golden set, not just whichever happens to be sampled."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT DISTINCT ON (source_pdf) source_pdf, section, page_start, page_end, domain, text "
        "FROM text_parents ORDER BY source_pdf, length(text) DESC"
    ).fetchall()
    conn.close()
    columns = ["source_pdf", "section", "page_start", "page_end", "domain", "text"]
    return [dict(zip(columns, row)) for row in rows]


def load_completed_source_pdfs():
    """Reads any golden_set.jsonl already on disk so a rerun (e.g. after a
    Gemini free-tier daily quota reset) skips papers already processed
    instead of regenerating them and burning quota twice."""
    if not os.path.exists(GOLDEN_SET_PATH):
        return set()
    completed = set()
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                completed.add(json.loads(line)["source_pdf"])
    return completed


def _write_goldens(goldens, chunk):
    os.makedirs(os.path.dirname(GOLDEN_SET_PATH), exist_ok=True)
    with open(GOLDEN_SET_PATH, "a", encoding="utf-8") as f:
        for golden in goldens:
            record = {
                "input": golden.input,
                "expected_output": golden.expected_output,
                "context": golden.context,
                "source_pdf": chunk["source_pdf"],
                "domain": chunk["domain"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_golden_set(chunks=None, synthesizer=None):
    """Processes one paper (one parent chunk) at a time and appends results
    to GOLDEN_SET_PATH immediately, so hitting Gemini's free-tier daily
    quota mid-run loses at most the current paper's progress - not
    everything - and a rerun the next day resumes from where it left off."""
    if chunks is None:
        chunks = load_representative_chunks()
    if synthesizer is None:
        synthesizer = Synthesizer(model=GeminiDeepEvalModel())

    completed = load_completed_source_pdfs()
    remaining = [c for c in chunks if c["source_pdf"] not in completed]

    if not remaining:
        print("All papers already have golden examples - nothing to do.")
        return []

    print(f"{len(completed)} paper(s) already done, {len(remaining)} remaining.")

    all_new_goldens = []
    papers_processed = 0
    for chunk in remaining:
        try:
            goldens = synthesizer.generate_goldens_from_contexts(
                contexts=[[chunk["text"]]],
                max_goldens_per_context=MAX_GOLDENS_PER_CONTEXT,
                source_files=[chunk["source_pdf"]],
            )
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(
                    f"Gemini quota exhausted after {len(all_new_goldens)} new golden(s) "
                    f"this run. Progress saved - rerun tomorrow to continue."
                )
                break
            raise

        _write_goldens(goldens, chunk)
        all_new_goldens.extend(goldens)
        papers_processed += 1
        print(f"  {chunk['source_pdf']}: +{len(goldens)} golden(s)")

    print(f"Wrote {len(all_new_goldens)} new golden examples to {GOLDEN_SET_PATH}")
    if all_new_goldens:
        print(f"({papers_processed} paper(s) processed this run - remember to `dvc add {GOLDEN_SET_PATH}` "
              f"and commit to version this update.)")

    return all_new_goldens


if __name__ == "__main__":
    generate_golden_set()
