from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.chunking_config import (
    CHUNKING_VERSION,
    TOKENIZER_ENCODING,
    PARENT_MAX_TOKENS,
    CHILD_MAX_TOKENS,
    PARENT_OVERLAP_TOKENS,
    CHILD_OVERLAP_TOKENS,
)


ENCODING = tiktoken.get_encoding(TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def split_text(text: str, max_tokens: int, overlap_tokens: int) -> list:
    """Token-aware split via LangChain's RecursiveCharacterTextSplitter
    (paragraph -> sentence -> word -> char, whichever level fits)."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER_ENCODING,
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
    )
    return splitter.split_text(text)


def chunk_document(source_pdf: str, sections: list, stem: str = None) -> dict:
    """Cuts ONE document's sections into parent chunks (what the model reads) holding child chunks (what is
    searched). Chunk ids are `<stem>_p<n>` and `<stem>_c<n>`; the stem defaults to the file name without its
    extension. Returns {"source_pdf", "chunking_version", "parents": [...]}."""
    stem = stem or Path(source_pdf).stem
    parent_chunks = []
    parent_counter = 0
    child_counter = 0

    for section in sections:
        if not section["text"].strip():
            continue

        parent_texts = split_text(section["text"], PARENT_MAX_TOKENS, PARENT_OVERLAP_TOKENS)

        for parent_text in parent_texts:
            parent_counter += 1
            parent_id = f"{stem}_p{parent_counter}"

            child_texts = split_text(parent_text, CHILD_MAX_TOKENS, CHILD_OVERLAP_TOKENS)

            children = []
            for child_text in child_texts:
                child_counter += 1
                children.append({
                    "chunk_id": f"{stem}_c{child_counter}",
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "source_pdf": source_pdf,
                    "section": section["heading"],
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                    "text": child_text,
                })

            parent_chunks.append({
                "chunk_id": parent_id,
                "chunk_type": "parent",
                "source_pdf": source_pdf,
                "section": section["heading"],
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "text": parent_text,
                "children": children,
            })

    return {"source_pdf": source_pdf, "chunking_version": CHUNKING_VERSION, "parents": parent_chunks}


