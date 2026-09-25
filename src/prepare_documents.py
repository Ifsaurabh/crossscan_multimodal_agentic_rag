import json
import re
from pathlib import Path

TEXT_DIR = Path(__file__).parent.parent / "data" / "text"
PREPARED_DIR = Path(__file__).parent.parent / "data" / "prepared"

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


def prepare_documents():
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(TEXT_DIR.glob("*.json"))
    print(f"Found {len(json_files)} text files to prepare")

    for text_path in json_files:
        doc = json.loads(text_path.read_text(encoding="utf-8"))
        sections, tables = split_into_sections(doc["blocks"])

        kept_sections = []
        dropped_count = 0
        for section in sections:
            if is_references_heading(section["heading"]):
                dropped_count += 1
                continue
            if not section["text"]:
                continue
            kept_sections.append(section)

        # A table's own section never goes through the section-drop loop above
        # (tables are tracked separately), so references-section tables need
        # their own check against the same heading list.
        kept_tables = [t for t in tables if not is_references_heading(t["section_heading"])]

        prepared = {
            "source_pdf": doc["source_pdf"],
            "sections": kept_sections,
            "tables": kept_tables,
        }

        out_path = PREPARED_DIR / text_path.name
        out_path.write_text(json.dumps(prepared, indent=2), encoding="utf-8")

        tables_dropped = len(tables) - len(kept_tables)
        print(f" - {text_path.stem}: {len(kept_sections)} sections kept, {dropped_count} references section(s) dropped, "
              f"{len(kept_tables)} tables kept, {tables_dropped} reference-section table(s) dropped")

    print(f"\nPrepared documents saved to {PREPARED_DIR}")


if __name__ == "__main__":
    prepare_documents()
