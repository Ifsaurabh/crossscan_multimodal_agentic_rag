import json
import re
from pathlib import Path

from query_guardrail import INJECTION_REGEX

PREPARED_DIR = Path(__file__).parent.parent / "data" / "prepared"
GUARDED_DIR = Path(__file__).parent.parent / "data" / "guarded"
REPORT_PATH = Path(__file__).parent.parent / "data" / "guardrail_report.json"
DOMAIN_MAP_PATH = Path(__file__).parent.parent / "data" / "domain_classification.json"

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

    domain_map = {d["source_pdf"]: d["domain"] for d in json.loads(DOMAIN_MAP_PATH.read_text(encoding="utf-8"))} \
        if DOMAIN_MAP_PATH.exists() else {}

    report = {"documents": []}

    for doc_path in prepared_files:
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        domain = domain_map.get(doc["source_pdf"], "unclassified")
        doc_entry = {
            "source_pdf": doc["source_pdf"],
            "domain": domain,
            "pii_redactions": 0,
            "pii_locations": [],
            "sections_dropped": [],
            "tables_dropped": [],
            "injection_flags": [],  # flagged only, never blocks/drops content
        }

        def guard(text, where):
            unsafe = contains_unsafe_content(text)
            if unsafe:
                return None, unsafe

            injection_match = INJECTION_REGEX.search(text)
            if injection_match:
                doc_entry["injection_flags"].append(
                    {"where": where, "domain": domain, "matched_text": injection_match.group()})

            redacted, count = redact_pii(text)
            doc_entry["pii_redactions"] += count
            if count:
                doc_entry["pii_locations"].append({"where": where, "domain": domain, "count": count})
            return redacted, None

        kept_sections = []
        for section in doc["sections"]:
            redacted, unsafe = guard(section["text"], section["heading"])
            if unsafe:
                doc_entry["sections_dropped"].append(
                    {"heading": section["heading"], "domain": domain, "matched_keywords": unsafe})
                continue
            section["text"] = redacted
            kept_sections.append(section)
        doc["sections"] = kept_sections

        # Tables bypass the section loop above (tracked separately, not glued
        # into section text) so they need the same checks here.
        kept_tables = []
        for table in doc.get("tables", []):
            redacted, unsafe = guard(table["text"], table["section_heading"])
            if unsafe:
                doc_entry["tables_dropped"].append(
                    {"section_heading": table["section_heading"], "domain": domain, "matched_keywords": unsafe})
                continue
            table["text"] = redacted
            kept_tables.append(table)
        doc["tables"] = kept_tables

        out_path = GUARDED_DIR / doc_path.name
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

        report["documents"].append(doc_entry)
        print(f" - {doc_path.stem} [{domain}]: {doc_entry['pii_redactions']} PII redactions, "
              f"{len(doc_entry['sections_dropped'])} section(s) + {len(doc_entry['tables_dropped'])} table(s) "
              f"dropped for unsafe content, {len(doc_entry['injection_flags'])} injection flag(s)")

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nGuarded documents saved to {GUARDED_DIR}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    run_guardrails()
