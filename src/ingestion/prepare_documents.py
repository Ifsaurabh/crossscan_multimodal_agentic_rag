import re

REFERENCES_HEADINGS = {
    "references", "bibliography", "works cited", "reference list",
}

NUMBERING_PREFIX = re.compile(r"^\s*\d+(\.\d+)*\.?\s+")


def is_references_heading(heading: str) -> bool:
    if not heading:
        return False
    normalized = NUMBERING_PREFIX.sub("", heading.strip()).lower().rstrip(":")
    return normalized in REFERENCES_HEADINGS


def split_into_sections(blocks):
    sections = []
    tables = []
    current = None

    for block in blocks:
        if block["label"] == "section_header":
            if current:
                sections.append(current)
            current = {
                "heading": block["text"],
                "level": block["level"],
                "text": "",
                "pages": [],
            }
            if block["page"] is not None:
                current["pages"].append(block["page"])
            continue

        if current is None:
            current = {"heading": None, "level": 0, "text": "", "pages": []}

        # Tables are pulled out to their own list (destined for graph nodes,
        # like images) instead of being glued into prose text.
        if block["label"] == "table":
            tables.append({
                "text": block["text"],
                "caption": block.get("caption") or "",
                "page": block["page"],
                "section_heading": current["heading"],
            })
        else:
            current["text"] += (block["text"] + "\n\n")

        if block["page"] is not None:
            current["pages"].append(block["page"])

    if current:
        sections.append(current)

    for section in sections:
        section["text"] = section["text"].strip()
        pages = section.pop("pages")
        section["page_start"] = min(pages) if pages else None
        section["page_end"] = max(pages) if pages else None

    return sections, tables


def prepare_document(blocks: list) -> dict:
    """The sections and tables of ONE document, ready for the guardrails and chunking.

    References sections and empty sections are dropped. A table's own section never goes through the
    section-drop loop (tables are tracked separately), so tables under a references heading get their own
    check against the same heading list."""
    sections, tables = split_into_sections(blocks)

    kept_sections, references_dropped = [], 0
    for section in sections:
        if is_references_heading(section["heading"]):
            references_dropped += 1
            continue
        if not section["text"]:
            continue
        kept_sections.append(section)

    kept_tables = [t for t in tables if not is_references_heading(t["section_heading"])]
    return {
        "sections": kept_sections,
        "tables": kept_tables,
        "references_dropped": references_dropped,
        "reference_tables_dropped": len(tables) - len(kept_tables),
    }


