import json
from pathlib import Path

from docling.document_converter import DocumentConverter

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
TEXT_DIR = Path(__file__).parent.parent / "data" / "text"

TEXT_LABELS = {"text", "section_header", "footnote", "list_item", "caption", "title"}


def extract_blocks(document):
    blocks = []
    for item, _ in document.iterate_items():
        label = getattr(item, "label", None)
        text = getattr(item, "text", None)

        if label not in TEXT_LABELS or not text:
            continue

        prov = getattr(item, "prov", None)
        page = prov[0].page_no if prov else None
        level = getattr(item, "level", None) if label == "section_header" else None

        blocks.append({
            "label": str(label),
            "level": level,
            "page": page,
            "text": text,
        })

    return blocks


def extract_text():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process")

    for pdf_path in pdf_files:
        print(f"Extracting text: {pdf_path.name}")
        result = converter.convert(str(pdf_path))
        blocks = extract_blocks(result.document)

        out_path = TEXT_DIR / (pdf_path.stem + ".json")
        out_path.write_text(
            json.dumps({"source_pdf": pdf_path.name, "blocks": blocks}, indent=2),
            encoding="utf-8",
        )
        print(f" -> saved {out_path.name} ({len(blocks)} blocks)")


if __name__ == "__main__":
    extract_text()
