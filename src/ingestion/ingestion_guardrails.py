"""Ingestion guardrail.

PII is redacted by Presidio and injection is scored by Llama Prompt Guard (both
engines live in query_guardrail). Injection is FLAG-ONLY here: the AI-security
paper in the corpus legitimately quotes attack phrases, so a flagged section is
reported (with its score) but never dropped.
"""
from shared import query_guardrail as qg


# Small representative unsafe-content keyword list (generic; not domain-specific).
UNSAFE_KEYWORDS = [
    "kill yourself", "child sexual", "how to make a bomb", "genocide propaganda",
]


def redact_pii(text: str):
    return qg.redact_pii(text)


def contains_unsafe_content(text: str):
    lowered = text.lower()
    return [kw for kw in UNSAFE_KEYWORDS if kw in lowered]


def guard_document(doc: dict, domain: str = "unclassified"):
    """Guards ONE prepared document ({"source_pdf", "sections", "tables"}): redacts personal data, drops a section
    or table that holds unsafe content, and flags (never drops) text that looks like an injection.
    Returns (the guarded document, its report entry)."""
    doc_entry = {
        "source_pdf": doc["source_pdf"],
        "domain": domain,
        "pii_redactions": 0,
        "pii_locations": [],
        "sections_dropped": [],
        "tables_dropped": [],
        "injection_flags": [],  # flagged only, never blocks/drops content
    }

    # Prompt Guard scores every section and table together, in batches (not one text at a time).
    section_count = len(doc["sections"])
    scores = qg.injection_scores([s["text"] for s in doc["sections"]] + [t["text"] for t in doc.get("tables", [])])

    def guard(text, where, score):
        unsafe = contains_unsafe_content(text)
        if unsafe:
            return None, unsafe

        injection = qg.classify_score(score)
        if injection["flagged"]:
            doc_entry["injection_flags"].append(
                {"where": where, "domain": domain, "score": round(injection["score"], 4)})

        redacted, count = redact_pii(text)
        doc_entry["pii_redactions"] += count
        if count:
            doc_entry["pii_locations"].append({"where": where, "domain": domain, "count": count})
        return redacted, None

    kept_sections = []
    for position, section in enumerate(doc["sections"]):
        redacted, unsafe = guard(section["text"], section["heading"], scores[position])
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
    for position, table in enumerate(doc.get("tables", [])):
        redacted, unsafe = guard(table["text"], table["section_heading"], scores[section_count + position])
        caption = table.get("caption") or ""
        if caption and not unsafe:
            unsafe = contains_unsafe_content(caption)
            if not unsafe:
                caption, caption_count = redact_pii(caption)
                doc_entry["pii_redactions"] += caption_count
                if caption_count:
                    doc_entry["pii_locations"].append(
                        {"where": table["section_heading"], "domain": domain, "count": caption_count})
        if unsafe:
            doc_entry["tables_dropped"].append(
                {"section_heading": table["section_heading"], "domain": domain, "matched_keywords": unsafe})
            continue
        table["text"] = redacted
        table["caption"] = caption
        kept_tables.append(table)
    doc["tables"] = kept_tables
    return doc, doc_entry


