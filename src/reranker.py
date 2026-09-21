from sentence_transformers import CrossEncoder

from retrieval_config import RERANKER_MODEL

_model = None


def get_reranker_model():
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query: str, candidates: list, top_k: int = None) -> list:
    """Re-scores candidates (dicts with a 'text' key) by relevance to the
    query using a cross-encoder. Returns candidates sorted by rerank score,
    with a 'rerank_score' field added."""
    if not candidates:
        return []

    model = get_reranker_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    return scored[:top_k] if top_k else scored
