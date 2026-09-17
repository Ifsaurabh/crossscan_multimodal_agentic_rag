import json
import re
from pathlib import Path

import pymupdf
from PIL import Image

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
TEXT_DIR = Path(__file__).parent.parent / "data" / "text"
IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"
REPORT_PATH = Path(__file__).parent.parent / "data" / "quality_report.json"

MIN_CHARS_PER_PAGE = 200
MIN_IMAGE_DIM = 50
GARBAGE_CHAR_RATIO_THRESHOLD = 0.05

GARBAGE_PATTERN = re.compile(r"[^\w\s.,;:!?()\-'\"/%]")


def check_text_files():
    results = []
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        stem = pdf_path.stem
        text_path = TEXT_DIR / f"{stem}.json"

        entry = {"file": stem, "issues": []}

        if not text_path.exists():
            entry["issues"].append("missing_text_file")
            results.append(entry)
            continue

        doc = json.loads(text_path.read_text(encoding="utf-8"))
        blocks = doc["blocks"]
        text = "\n\n".join(b["text"] for b in blocks)
        char_count = len(text)
        page_count = len(pymupdf.open(pdf_path))
        chars_per_page = char_count / page_count if page_count else 0
        heading_count = sum(1 for b in blocks if b["label"] == "section_header")
        garbage_chars = len(GARBAGE_PATTERN.findall(text))
        garbage_ratio = garbage_chars / char_count if char_count else 1.0

        entry.update({
            "page_count": page_count,
            "char_count": char_count,
            "chars_per_page": round(chars_per_page, 1),
            "heading_count": heading_count,
            "garbage_ratio": round(garbage_ratio, 4),
        })

        if chars_per_page < MIN_CHARS_PER_PAGE:
            entry["issues"].append("low_chars_per_page")
        if heading_count == 0:
            entry["issues"].append("no_headings_found")
        if garbage_ratio > GARBAGE_CHAR_RATIO_THRESHOLD:
            entry["issues"].append("high_garbage_char_ratio")

        results.append(entry)

    return results


def check_images():
    metadata_path = IMAGES_DIR / "metadata.json"
    if not metadata_path.exists():
        return []

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    flagged = []
    kept = []

    for item in metadata:
        image_path = IMAGES_DIR / item["image_file"]
        entry = {"file": item["image_file"], "source_pdf": item["source_pdf"], "issues": []}

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                entry["width"] = width
                entry["height"] = height
                if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
                    entry["issues"].append("tiny_image_likely_icon_or_logo")
        except Exception as e:
            entry["issues"].append(f"corrupt_or_unreadable: {e}")

        if entry["issues"]:
            flagged.append(entry)
        else:
            kept.append(item)

    for entry in flagged:
        image_path = IMAGES_DIR / entry["file"]
        if image_path.exists():
            image_path.unlink()

    metadata_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")

    return flagged


def run_quality_check():
    print("Checking text extraction quality...")
    text_results = check_text_files()

    print("Checking image extraction quality...")
    image_issues = check_images()

    text_flagged = [r for r in text_results if r["issues"]]

    report = {
        "text_files_checked": len(text_results),
        "text_files_flagged": len(text_flagged),
        "text_details": text_results,
        "image_files_flagged": len(image_issues),
        "image_details": image_issues,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{len(text_flagged)}/{len(text_results)} text files flagged:")
    for r in text_flagged:
        print(f" - {r['file']}: {r['issues']}")

    print(f"\n{len(image_issues)} image files flagged:")
    for r in image_issues[:20]:
        print(f" - {r['file']} ({r['source_pdf']}): {r['issues']}")
    if len(image_issues) > 20:
        print(f" ... and {len(image_issues) - 20} more")

    print(f"\nFull report saved to {REPORT_PATH}")


if __name__ == "__main__":
    run_quality_check()
