from sentence_transformers import SentenceTransformer

from shared.embedding_config import TEXT_MODEL_NAME


def child_embedding_text(child: dict) -> str:
    """Structure-aware embedding: prepend document/section context so the embedding vector itself is
    distinguishable, not just the metadata stored alongside it. Improves matching precision for short/generic
    chunks that read almost identically across different papers/sections."""
    return f"Document: {child['source_pdf']} | Section: {child['section']}\n\n{child['text']}"


def load_text_model():
    return SentenceTransformer(TEXT_MODEL_NAME)


def embed_texts(model, texts: list):
    """One normalised embedding per text (the same call the chunk embedding uses)."""
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


