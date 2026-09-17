import json
import re
from pathlib import Path

PREPARED_DIR = Path(__file__).parent.parent / "data" / "prepared"
GUARDED_DIR = Path(__file__).parent.parent / "data" / "guarded"
REPORT_PATH = Path(__file__).parent.parent / "data" / "guardrail_report.json"

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Small representative unsafe-content keyword list (generic; not domain-specific).
UNSAFE_KEYWORDS = [
    "kill yourself", "child sexual", "how to make a bomb", "genocide propaganda",
]


def redact_pii(text: str):
    redactions = 0

    def replace_email(match):
        nonlocal redactions
        redactions += 1
        return "[REDACTED_EMAIL]"

    text = EMAIL_PATTERN.sub(replace_email, text)

    return text, redactions


def contains_unsafe_content(text: str):
    lowered = text.lower()
    return [kw for kw in UNSAFE_KEYWORDS if kw in lowered]


def run_guardrails():
    GUARDED_DIR.mkdir(parents=True, exist_ok=True)

    prepared_files = sorted(PREPARED_DIR.glob("*.json"))
    print(f"Found {len(prepared_files)} prepared documents to guard")

    report = {"documents": []}

    for doc_path in prepared_files:
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        doc_entry = {"source_pdf": doc["source_pdf"], "pii_redactions": 0, "sections_dropped": []}

        kept_sections = []
        for section in doc["sections"]:
            unsafe_hits = contains_unsafe_content(section["text"])
            if unsafe_hits:
                doc_entry["sections_dropped"].append({
                    "heading": section["heading"],
                    "matched_keywords": unsafe_hits,
                })
                continue

            redacted_text, redaction_count = redact_pii(section["text"])
            section["text"] = redacted_text
            doc_entry["pii_redactions"] += redaction_count
            kept_sections.append(section)

        doc["sections"] = kept_sections

        out_path = GUARDED_DIR / doc_path.name
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

        report["documents"].append(doc_entry)
        print(f" - {doc_path.stem}: {doc_entry['pii_redactions']} PII redactions, "
              f"{len(doc_entry['sections_dropped'])} section(s) dropped for unsafe content")

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nGuarded documents saved to {GUARDED_DIR}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    run_guardrails()
