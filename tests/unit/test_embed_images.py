import torch
from PIL import Image

from ingestion import embed_images as ei


class FakeProcessor:
    def __init__(self):
        self.batch_sizes, self.modes = [], []

    def __call__(self, images, return_tensors="pt"):
        self.batch_sizes.append(len(images))
        self.modes.extend(img.mode for img in images)
        return {"pixel_values": torch.zeros(len(images), 3, 4, 4)}


class FakeVisionOutputs:
    def __init__(self, n):
        self.pooler_output = torch.ones(n, 2)


class FakeModel:
    def vision_model(self, pixel_values):
        return FakeVisionOutputs(pixel_values.shape[0])

    def visual_projection(self, pooled):
        return torch.tensor([[3.0, 4.0]] * pooled.shape[0])


def test_a_picture_is_embedded_to_a_unit_length_vector():
    vector = ei.embed_pil_image(FakeModel(), FakeProcessor(), Image.new("RGB", (50, 50)))

    assert len(vector) == 2
    assert abs(sum(v ** 2 for v in vector) ** 0.5 - 1.0) < 1e-5


def test_many_pictures_are_embedded_in_batches_in_the_order_given():
    processor = FakeProcessor()

    vectors = ei.embed_pil_images(FakeModel(), processor, [Image.new("RGB", (8, 8)) for _ in range(20)], batch_size=8)

    assert processor.batch_sizes == [8, 8, 4] and len(vectors) == 20
    assert all(abs(sum(v ** 2 for v in vec) ** 0.5 - 1.0) < 1e-5 for vec in vectors)


def test_no_pictures_gives_no_vectors_and_no_model_call():
    processor = FakeProcessor()
    assert ei.embed_pil_images(FakeModel(), processor, []) == [] and processor.batch_sizes == []


def test_the_default_batch_is_sixteen_pictures():
    assert ei.IMAGE_BATCH_SIZE == 16


def test_a_picture_with_an_alpha_channel_is_converted_before_embedding():
    processor = FakeProcessor()
    ei.embed_pil_images(FakeModel(), processor, [Image.new("RGBA", (10, 10)), Image.new("L", (10, 10))])
    assert processor.modes == ["RGB", "RGB"]
