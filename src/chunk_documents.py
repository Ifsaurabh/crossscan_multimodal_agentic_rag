import json
import time
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking_config import (
    CHUNKING_VERSION,
    TOKENIZER_ENCODING,
    PARENT_MAX_TOKENS,
    CHILD_MAX_TOKENS,
    PARENT_OVERLAP_TOKENS,
    CHILD_OVERLAP_TOKENS,
)

GUARDED_DIR = Path(__file__).parent.parent / "data" / "guarded"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
REPORT_PATH = Path(__file__).parent.parent / "data" / "chunking_report.json"

ENCODING = tiktoken.get_encoding(TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def split_oversized_unit(text: str, max_tokens: int, overlap_tokens: int):
    """Fallback for a single paragraph/unit that exceeds the cap on its own.
    Uses RecursiveCharacterTextSplitter (paragraph -> sentence -> word -> char),
    token-aware via tiktoken, instead of the custom parent-child grouping logic."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER_ENCODING,
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
    )
    return splitter.split_text(text)


def group_units(units, max_tokens: int, overlap_tokens: int):
    """Greedily group text units (paragraphs or sentences) into chunks under max_tokens,
    carrying forward trailing units worth ~overlap_tokens into the next chunk."""
    unit_tokens = [(u, count_tokens(u)) for u in units]

    chunks = []
    current = []
    current_tokens = 0

    for unit, tokens in unit_tokens:
        if tokens > max_tokens:
            if current:
                chunks.append(current)
                current = []
                current_tokens = 0
            for sub in split_oversized_unit(unit, max_tokens, overlap_tokens):
                chunks.append([(sub, count_tokens(sub))])
            continue

        if current_tokens + tokens > max_tokens and current:
            chunks.append(current)
            overlap_units = []
            overlap_count = 0
            for u, t in reversed(current):
                if overlap_count + t > overlap_tokens:
                    break
                overlap_units.insert(0, (u, t))
                overlap_count += t
            current = overlap_units
            current_tokens = overlap_count

        current.append((unit, tokens))
        current_tokens += tokens

    if current:
        chunks.append(current)

    return ["\n\n".join(u for u, _ in c) for c in chunks]


def chunk_documents():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    guarded_files = sorted(GUARDED_DIR.glob("*.json"))
    print(f"Found {len(guarded_files)} guarded documents to chunk")

    report = {
        "chunking_version": CHUNKING_VERSION,
        "documents": [],
    }

    for doc_path in guarded_files:
        start_time = time.perf_counter()
        doc = json.loads(doc_path.read_text(encoding="utf-8"))

        parent_chunks = []
        child_token_counts = []
        parent_token_counts = []
        parent_counter = 0
        child_counter = 0

        for section in doc["sections"]:
            paragraphs = [p.strip() for p in section["text"].split("\n\n") if p.strip()]
            if not paragraphs:
                continue

            parent_texts = group_units(paragraphs, PARENT_MAX_TOKENS, PARENT_OVERLAP_TOKENS)

            for parent_text in parent_texts:
                parent_counter += 1
                parent_id = f"{doc_path.stem}_p{parent_counter}"
                parent_tokens = count_tokens(parent_text)
                parent_token_counts.append(parent_tokens)

                parent_paragraphs = [p.strip() for p in parent_text.split("\n\n") if p.strip()]
                child_texts = group_units(parent_paragraphs, CHILD_MAX_TOKENS, CHILD_OVERLAP_TOKENS)

                children = []
                for child_text in child_texts:
                    child_counter += 1
                    child_id = f"{doc_path.stem}_c{child_counter}"
                    child_tokens = count_tokens(child_text)
                    child_token_counts.append(child_tokens)

                    children.append({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "chunk_type": "child",
                        "source_pdf": doc["source_pdf"],
                        "section": section["heading"],
                        "page_start": section.get("page_start"),
                        "page_end": section.get("page_end"),
                        "text": child_text,
                    })

                parent_chunks.append({
                    "chunk_id": parent_id,
                    "chunk_type": "parent",
                    "source_pdf": doc["source_pdf"],
                    "section": section["heading"],
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                    "text": parent_text,
                    "children": children,
                })

        elapsed = time.perf_counter() - start_time

        out_path = CHUNKS_DIR / doc_path.name
        out_path.write_text(
            json.dumps({
                "source_pdf": doc["source_pdf"],
                "chunking_version": CHUNKING_VERSION,
                "parents": parent_chunks,
            }, indent=2),
            encoding="utf-8",
        )

        doc_report = {
            "source_pdf": doc["source_pdf"],
            "parent_count": len(parent_token_counts),
            "child_count": len(child_token_counts),
            "parent_tokens": {
                "min": min(parent_token_counts) if parent_token_counts else 0,
                "max": max(parent_token_counts) if parent_token_counts else 0,
                "avg": round(sum(parent_token_counts) / len(parent_token_counts), 1) if parent_token_counts else 0,
            },
            "child_tokens": {
                "min": min(child_token_counts) if child_token_counts else 0,
                "max": max(child_token_counts) if child_token_counts else 0,
                "avg": round(sum(child_token_counts) / len(child_token_counts), 1) if child_token_counts else 0,
            },
            "latency_seconds": round(elapsed, 3),
        }
        report["documents"].append(doc_report)

        print(f" - {doc_path.stem}: {doc_report['parent_count']} parents, "
              f"{doc_report['child_count']} children, {doc_report['latency_seconds']}s")

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nChunks saved to {CHUNKS_DIR}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    chunk_documents()
