import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

from embedding_config import TEXT_MODEL_NAME

TEXT_EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "embeddings" / "text"
TEXT_DIR = Path(__file__).parent.parent / "data" / "text"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "domain_classification.json"
REPORT_PATH = Path(__file__).parent.parent / "data" / "domain_classification_report.json"

SEED_DOMAIN_LABELS = {
    "lung-cancer": "This document is about lung cancer and medical imaging, such as CT scans, tumor detection, and cancer diagnosis.",
    "land-cover": "This document is about land cover and remote sensing, such as satellite imagery, vegetation classification, and environmental monitoring.",
}

NEW_DOMAIN_THRESHOLD = 0.60

# Journal front-matter labels that sometimes get tagged as section headers
# before the real title (e.g. "OPEN ACCESS", "REVIEWED BY") - skip these
# when looking for the actual document title.
FRONT_MATTER_LABELS = {
    "open access", "reviewed by", "edited by", "citation", "copyright",
    "correspondence", "*correspondence", "received", "accepted", "published",
}


def slugify_title(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return "-".join(words[:5]) if words else "unknown-domain"


def get_document_title(doc_path: Path) -> str:
    text_path = TEXT_DIR / doc_path.name
    if text_path.exists():
        blocks = json.loads(text_path.read_text(encoding="utf-8"))["blocks"]
        for b in blocks:
            if b["label"] == "section_header" and b["text"].strip().lower() not in FRONT_MATTER_LABELS:
                return b["text"]
    return doc_path.stem


def classify_domain():
    print(f"Loading text embedding model: {TEXT_MODEL_NAME}")
    model = SentenceTransformer(TEXT_MODEL_NAME)

    domain_names = list(SEED_DOMAIN_LABELS.keys())
    domain_embeddings = list(model.encode(list(SEED_DOMAIN_LABELS.values()), normalize_embeddings=True))

    doc_files = sorted(TEXT_EMBEDDINGS_DIR.glob("*.json"))
    print(f"Found {len(doc_files)} documents to classify")

    results = {}
    report = {"threshold": NEW_DOMAIN_THRESHOLD, "documents": []}

    for doc_path in doc_files:
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        records = doc["records"]
        if not records:
            continue

        chunk_vectors = [r["embedding"] for r in records]
        doc_vector = [sum(col) / len(col) for col in zip(*chunk_vectors)]

        scores = util.cos_sim(doc_vector, domain_embeddings)[0]
        best_index = int(scores.argmax())
        best_score = float(scores[best_index])

        if best_score >= NEW_DOMAIN_THRESHOLD:
            domain = domain_names[best_index]
            created_new = False
        else:
            title = get_document_title(doc_path)
            domain = slugify_title(title)
            domain_names.append(domain)
            domain_embeddings.append(model.encode(title, normalize_embeddings=True))
            created_new = True

        results[doc["source_pdf"]] = domain
        report["documents"].append({
            "source_pdf": doc["source_pdf"],
            "domain": domain,
            "best_seed_score": round(best_score, 4),
            "created_new_domain": created_new,
        })

        flag = " [NEW DOMAIN]" if created_new else ""
        print(f" - {doc['source_pdf']}: {domain} (score {round(best_score, 4)}){flag}")

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report["discovered_domains"] = domain_names
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nDomains: {domain_names}")
    print(f"Domain mapping saved to {OUTPUT_PATH}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    classify_domain()
