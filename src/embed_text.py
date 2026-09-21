import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from embedding_config import EMBEDDING_VERSION, TEXT_MODEL_NAME

CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "embeddings" / "text"
REPORT_PATH = Path(__file__).parent.parent / "data" / "text_embedding_report.json"


def embed_text():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading text embedding model: {TEXT_MODEL_NAME}")
    model = SentenceTransformer(TEXT_MODEL_NAME)

    chunk_files = sorted(CHUNKS_DIR.glob("*.json"))
    print(f"Found {len(chunk_files)} chunked documents")

    report = {
        "embedding_version": EMBEDDING_VERSION,
        "model": TEXT_MODEL_NAME,
        "documents": [],
    }

    total_children = 0

    for chunk_path in chunk_files:
        start_time = time.perf_counter()
        doc = json.loads(chunk_path.read_text(encoding="utf-8"))

        children = []
        for parent in doc["parents"]:
            children.extend(parent["children"])

        if not children:
            continue

        # Structure-aware embedding: prepend document/section context so the
        # embedding vector itself is distinguishable, not just the metadata
        # stored alongside it. Improves matching precision for short/generic
        # chunks that read almost identically across different papers/sections.
        texts = [
            f"Document: {c['source_pdf']} | Section: {c['section']}\n\n{c['text']}"
            for c in children
        ]
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        records = []
        for child, vector in zip(children, vectors):
            records.append({
                "chunk_id": child["chunk_id"],
                "parent_id": child["parent_id"],
                "source_pdf": child["source_pdf"],
                "section": child["section"],
                "page_start": child["page_start"],
                "page_end": child["page_end"],
                "embedding": vector.tolist(),
            })

        elapsed = time.perf_counter() - start_time
        total_children += len(records)

        out_path = EMBEDDINGS_DIR / chunk_path.name
        out_path.write_text(
            json.dumps({
                "source_pdf": doc["source_pdf"],
                "embedding_version": EMBEDDING_VERSION,
                "embedding_dim": len(vectors[0]),
                "records": records,
            }, indent=2),
            encoding="utf-8",
        )

        report["documents"].append({
            "source_pdf": doc["source_pdf"],
            "chunks_embedded": len(records),
            "latency_seconds": round(elapsed, 3),
        })

        print(f" - {chunk_path.stem}: {len(records)} child chunks embedded, {round(elapsed, 3)}s")

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nTotal chunks embedded: {total_children}")
    print(f"Embeddings saved to {EMBEDDINGS_DIR}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    embed_text()
