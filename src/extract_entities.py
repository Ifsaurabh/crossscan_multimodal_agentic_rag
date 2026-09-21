import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

load_dotenv()

PREPARED_DIR = Path(__file__).parent.parent / "data" / "prepared"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "entities.json"
REPORT_PATH = Path(__file__).parent.parent / "data" / "entity_extraction_report.json"

GEMINI_MODEL = "gemini-3.6-flash"

NUMBERING_PREFIX = re.compile(r"^\s*\d+(\.\d+)*\.?\s+")


def normalize_heading(heading: str) -> str:
    if not heading:
        return ""
    return NUMBERING_PREFIX.sub("", heading.strip()).lower()


def get_abstract_and_conclusion(sections):
    abstract = ""
    conclusion = ""
    for s in sections:
        heading = normalize_heading(s["heading"])
        if heading == "abstract" and not abstract:
            abstract = s["text"]
        elif "conclusion" in heading and not conclusion:
            conclusion = s["text"]
    return abstract, conclusion


def build_prompt(text):
    return f"""Extract entities from this research paper text, categorized into three types:
- METHODS: algorithms, models, or techniques used (e.g. CNN, YOLOv11, Discrete Wavelet Transform)
- DATASETS: named datasets used (e.g. LIDC-IDRI, Sentinel-2)
- METRICS: named evaluation metrics (e.g. accuracy, dice coefficient)

CRITICAL RULES:
- Only extract entities that are EXPLICITLY named in the text below, word for word.
- Do NOT infer, guess, or complete unstated specifics (e.g. if the text says "the other four YOLO versions" without naming them, do NOT invent version numbers).
- If a category has no explicitly named entities, return an empty list for it.
- Do NOT include entities from your general knowledge that are not literally present in this text.

Text:
{text}

Respond ONLY as JSON in this exact format, with no other text:
{{"methods": [...], "datasets": [...], "metrics": [...]}}"""


def query_gemini(client, prompt: str, max_retries=4) -> str:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return response.text.strip()
        except ServerError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"   (server busy, retrying in {wait}s...)")
            time.sleep(wait)


def parse_entities(raw_response: str):
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def verify_entities(entities, source_text):
    """Flag entities whose exact text does not appear in the source, so
    fabricated/inferred entities (the failure mode found with Qwen) can be
    caught and reviewed instead of silently trusted."""
    source_lower = source_text.lower()
    verified = {}
    unverified = {}

    for category, items in entities.items():
        verified[category] = []
        unverified[category] = []
        for item in items:
            if item.lower() in source_lower:
                verified[category].append(item)
            else:
                unverified[category].append(item)

    return verified, unverified


def extract_entities():
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

    doc_files = sorted(PREPARED_DIR.glob("*.json"))
    print(f"Found {len(doc_files)} documents")
    print(f"Using Gemini model: {GEMINI_MODEL}")

    # Only resume from existing output if it was produced by this same model -
    # otherwise stale data from a previous model/approach would be mistaken
    # for already-completed work.
    existing_report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else None
    if existing_report and existing_report.get("model") == GEMINI_MODEL:
        report = existing_report
        results = json.loads(OUTPUT_PATH.read_text(encoding="utf-8")) if OUTPUT_PATH.exists() else {}
    else:
        report = {"model": GEMINI_MODEL, "documents": []}
        results = {}
    already_done = {d["source_pdf"] for d in report["documents"]}

    for doc_path in doc_files:
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        source_pdf = doc["source_pdf"]

        if source_pdf in already_done:
            print(f" - {source_pdf}: already processed, skipping")
            continue

        abstract, conclusion = get_abstract_and_conclusion(doc["sections"])
        combined_text = f"{abstract}\n\n{conclusion}".strip()

        if not combined_text:
            print(f" - {source_pdf}: no abstract/conclusion found, skipped")
            report["documents"].append({"source_pdf": source_pdf, "skipped": True})
            REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            continue

        prompt = build_prompt(combined_text)
        start_time = time.perf_counter()
        raw_response = query_gemini(client, prompt)
        elapsed = time.perf_counter() - start_time

        entities = parse_entities(raw_response)
        if entities is None:
            print(f" - {source_pdf}: failed to parse JSON response ({round(elapsed, 1)}s)")
            report["documents"].append({
                "source_pdf": source_pdf, "parse_failed": True,
                "raw_response": raw_response, "latency_seconds": round(elapsed, 1),
            })
            REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            continue

        verified, unverified = verify_entities(entities, combined_text)
        results[source_pdf] = {"verified": verified, "unverified": unverified}

        total_unverified = sum(len(v) for v in unverified.values())
        report["documents"].append({
            "source_pdf": source_pdf,
            "methods_count": len(verified.get("methods", [])),
            "datasets_count": len(verified.get("datasets", [])),
            "metrics_count": len(verified.get("metrics", [])),
            "unverified_count": total_unverified,
            "unverified_items": unverified if total_unverified else None,
            "latency_seconds": round(elapsed, 1),
        })

        flag = f" [{total_unverified} UNVERIFIED - not found in source]" if total_unverified else ""
        print(f" - {source_pdf}: {len(verified.get('methods', []))} methods, "
              f"{len(verified.get('datasets', []))} datasets, "
              f"{len(verified.get('metrics', []))} metrics ({round(elapsed, 1)}s){flag}")

        # Save incrementally so a later failure doesn't lose completed work.
        OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nEntities saved to {OUTPUT_PATH}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    extract_entities()
