import reranker


class FakeCrossEncoder:
    def predict(self, pairs):
        # Score higher for pairs whose text contains "relevant"
        return [10.0 if "relevant" in text.lower() else 1.0 for _, text in pairs]


def test_rerank_sorts_by_relevance(monkeypatch):
    monkeypatch.setattr(reranker, "get_reranker_model", lambda: FakeCrossEncoder())

    candidates = [
        {"chunk_id": "c1", "text": "This is not what you want"},
        {"chunk_id": "c2", "text": "This is the relevant chunk"},
        {"chunk_id": "c3", "text": "Also irrelevant content"},
    ]

    result = reranker.rerank("query", candidates)

    assert result[0]["chunk_id"] == "c2"
    assert result[0]["rerank_score"] == 10.0


def test_rerank_respects_top_k(monkeypatch):
    monkeypatch.setattr(reranker, "get_reranker_model", lambda: FakeCrossEncoder())

    candidates = [{"chunk_id": f"c{i}", "text": "relevant text"} for i in range(5)]
    result = reranker.rerank("query", candidates, top_k=2)

    assert len(result) == 2


def test_rerank_handles_empty_candidates():
    result = reranker.rerank("query", [])
    assert result == []
