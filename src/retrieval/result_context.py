"""Turns the result of a graph run into the chunks and context strings that are scored or judged.
Used by the offline evaluation (evaluation/run_evaluation.py) and by the live scoring inside the app
(retrieval/online_eval.py), so it lives here and the app does not depend on the evaluation folder."""


def collect_chunks(result: dict) -> list:
    """Every chunk retrieved across all sub-queries of a graph run."""
    chunks = []
    for sq in result.get("sub_queries", []):
        chunks.extend(sq.get("chunks", []))
    return chunks


def collect_contexts(chunks: list) -> list:
    """Retrieved context strings (parent text when present), de-duplicated in
    order - the shape RAGAS/DeepEval expect for retrieval_context."""
    seen, contexts = set(), []
    for c in chunks:
        text = c.get("parent_text") or c.get("text") or ""
        if text and text not in seen:
            seen.add(text)
            contexts.append(text)
    return contexts
