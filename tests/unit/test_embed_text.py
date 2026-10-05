import numpy as np

from ingestion import embed_text as et


class FakeModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        self.seen.append((list(texts), show_progress_bar, normalize_embeddings))
        return np.array([[float(len(t)), 0.0, 1.0] for t in texts])


def test_a_child_is_embedded_with_its_document_and_section_in_front_of_its_text():
    text = et.child_embedding_text({"source_pdf": "sample.pdf", "section": "Results", "text": "The model achieved 98% accuracy."})

    assert text == "Document: sample.pdf | Section: Results\n\nThe model achieved 98% accuracy."


def test_the_texts_are_embedded_in_one_normalised_call_in_order():
    model = FakeModel()

    vectors = et.embed_texts(model, ["a", "bb", "ccc"])

    assert model.seen == [(["a", "bb", "ccc"], False, True)]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]
