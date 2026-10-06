"""injection_scores: the windows of many texts scored together in batches. A fake tokenizer and model; no real Prompt Guard."""
import pytest
import torch

from shared import query_guardrail as qg


class Tokenizer:
    """One token per word. Called with a string it gives its ids; called with a list it gives the batch the model receives."""

    def __call__(self, x, add_special_tokens=None, return_tensors=None, padding=None, truncation=None, max_length=None):
        if isinstance(x, str):
            return {"input_ids": list(range(len(x.split())))}
        return {"pieces": list(x)}

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


class Model:
    def __init__(self):
        self.batches = []

    def __call__(self, pieces):
        self.batches.append(len(pieces))
        logits = torch.tensor([[0.0, len(p.split()) / 100.0] for p in pieces])  # a longer window looks more suspicious
        return type("Out", (), {"logits": logits})()


@pytest.fixture
def guard(monkeypatch):
    model = Model()
    monkeypatch.setattr(qg, "_get_prompt_guard", lambda: (Tokenizer(), model))
    monkeypatch.setattr(qg, "injection_scores", _real_injection_scores)  # the conftest stub is replaced by the real function
    return model


_real_injection_scores = qg.injection_scores


def test_the_windows_of_all_the_texts_are_scored_together_in_batches(guard):
    long_text = " ".join(["word"] * 600)   # two windows of at most 510 tokens

    scores = qg.injection_scores(["a b c", long_text, "", "d e"], batch_size=2)

    assert guard.batches == [2, 2]                      # 4 windows (1 + 2 + 0 + 1), two per forward pass
    assert scores[2] == 0.0                             # an empty text scores 0 without a window
    assert scores[1] > scores[0] > 0 and scores[3] > 0  # the long text has the highest window


def test_a_text_scores_the_highest_of_its_windows(guard):
    text = " ".join(["word"] * 600)                      # windows of 510 and 90 words
    score = qg.injection_scores([text])[0]
    softmax_510 = torch.softmax(torch.tensor([0.0, 5.10]), dim=0)[1].item()
    assert score == pytest.approx(softmax_510, abs=1e-6)


def test_every_score_is_none_when_prompt_guard_is_not_available(monkeypatch):
    monkeypatch.setattr(qg, "_get_prompt_guard", lambda: None)
    monkeypatch.setattr(qg, "injection_scores", _real_injection_scores)
    assert qg.injection_scores(["a", "b"]) == [None, None]


def test_no_texts_means_no_model_call(guard):
    assert qg.injection_scores([]) == [] and guard.batches == []


def test_the_default_batch_is_sixteen_windows():
    assert qg.PROMPT_GUARD_BATCH == 16


def test_what_a_score_means():
    assert qg.classify_score(None) == {"score": None, "available": False, "flagged": False, "blocked": False}
    assert qg.classify_score(0.6)["flagged"] is True and qg.classify_score(0.6)["blocked"] is True   # flag and block are both 0.5
    assert qg.classify_score(0.99)["blocked"] is True and qg.classify_score(0.5)["blocked"] is True and qg.classify_score(0.49)["blocked"] is False and qg.classify_score(0.1)["flagged"] is False
