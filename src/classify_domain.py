import json
import re
from pathlib import Path

from llm_connection import generate

TEXT_DIR = Path(__file__).parent.parent / "data" / "text"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "domain_classification.json"


BATCH_SIZE = 100
SNIPPET_CHARS = 800

DOMAIN_SYSTEM_INSTRUCTION = (
    "You read raw text snippets taken from the start of research papers (may "
    "include journal/page noise before the real content - ignore that, focus on "
    "the actual paper). You will receive a numbered list of snippets. For each "
    "one, extract the paper's real title (ignore journal names, publisher names, "
    "page headers) and its research domain as a 2-4 word lowercase phrase (e.g. "
    "\"lung cancer imaging\", \"land cover remote sensing\", \"ai security\"). "
    "Respond with ONLY a JSON array of objects, same length and order as the "
    "input list: [{\"title\": ..., \"domain\": ...}, ...]. No other text, no code fences."
)


def slugify_title(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return "-".join(words[:5]) if words else "unknown-domain"


def get_document_snippet(doc_path: Path, max_chars: int = SNIPPET_CHARS) -> str:
    """Raw leading text (all block labels, in reading order) - classification input."""
    blocks = json.loads(doc_path.read_text(encoding="utf-8"))["blocks"]
    return " ".join(b["text"] for b in blocks)[:max_chars]


def classify_snippets_batch(snippets: list) -> list:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(snippets))
    result = generate(DOMAIN_SYSTEM_INSTRUCTION, numbered, tier="fast")

    raw = result.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)

    if len(parsed) != len(snippets):
        raise ValueError(f"Batch mismatch: {len(snippets)} snippets in, {len(parsed)} results back")

    return [{"title": p["title"].strip(), "domain": slugify_title(p["domain"])} for p in parsed]


def classify_domain():
    doc_files = sorted(TEXT_DIR.glob("*.json"))
    print(f"Found {len(doc_files)} documents to classify")

    docs = []
    for doc_path in doc_files:
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        docs.append({
            "source_pdf": doc["source_pdf"],
            "snippet": get_document_snippet(doc_path),
        })

    results = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        classified = classify_snippets_batch([d["snippet"] for d in batch])
        for d, c in zip(batch, classified):
            results.append({"source_pdf": d["source_pdf"], "title": c["title"], "domain": c["domain"]})
            print(f" - {d['source_pdf']}: {c['title']} -> {c['domain']}")

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDomain mapping saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    classify_domain()
