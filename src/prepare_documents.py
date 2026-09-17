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

    return sections


def prepare_documents():
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(TEXT_DIR.glob("*.json"))
    print(f"Found {len(json_files)} text files to prepare")

    for text_path in json_files:
        doc = json.loads(text_path.read_text(encoding="utf-8"))
        sections = split_into_sections(doc["blocks"])

        kept_sections = []
        dropped_count = 0
        for section in sections:
            if is_references_heading(section["heading"]):
                dropped_count += 1
                continue
            if not section["text"]:
                continue
            kept_sections.append(section)

        prepared = {
            "source_pdf": doc["source_pdf"],
            "sections": kept_sections,
        }

        out_path = PREPARED_DIR / text_path.name
        out_path.write_text(json.dumps(prepared, indent=2), encoding="utf-8")

        print(f" - {text_path.stem}: {len(kept_sections)} sections kept, {dropped_count} references section(s) dropped")

    print(f"\nPrepared documents saved to {PREPARED_DIR}")


if __name__ == "__main__":
    prepare_documents()
