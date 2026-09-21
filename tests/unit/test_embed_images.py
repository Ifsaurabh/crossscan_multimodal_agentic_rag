import json

import torch
from PIL import Image

import embed_images as ei
import embedding_config as cfg


class FakeProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(1, 3, 4, 4)}


class FakeVisionOutputs:
    pooler_output = torch.tensor([[1.0, 2.0]])


class FakeModel:
    def vision_model(self, pixel_values):
        return FakeVisionOutputs()

    def visual_projection(self, pooled):
        return torch.tensor([[3.0, 4.0]])


def test_embed_images_produces_normalized_embeddings(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    monkeypatch.setattr(ei, "IMAGES_DIR", images_dir)
    monkeypatch.setattr(ei, "METADATA_PATH", images_dir / "metadata.json")
    monkeypatch.setattr(ei, "EMBEDDINGS_PATH", tmp_path / "embeddings" / "image_embeddings.json")
    monkeypatch.setattr(ei, "REPORT_PATH", tmp_path / "image_embedding_report.json")

    img_path = images_dir / "fig1.png"
    Image.new("RGB", (50, 50)).save(img_path)

    metadata = [{"image_file": "fig1.png", "source_pdf": "a.pdf", "page": 2}]
    (images_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(ei, "CLIPModel", type("M", (), {"from_pretrained": staticmethod(lambda name: FakeModel())}))
    monkeypatch.setattr(ei, "CLIPProcessor", type("P", (), {"from_pretrained": staticmethod(lambda name: FakeProcessor())}))

    ei.embed_images()

    out_path = tmp_path / "embeddings" / "image_embeddings.json"
    result = json.loads(out_path.read_text(encoding="utf-8"))

    assert result["embedding_version"] == cfg.EMBEDDING_VERSION
    assert len(result["records"]) == 1
    rec = result["records"][0]
    assert rec["image_file"] == "fig1.png"
    assert rec["source_pdf"] == "a.pdf"
    assert rec["page"] == 2

    norm = sum(v ** 2 for v in rec["embedding"]) ** 0.5
    assert abs(norm - 1.0) < 1e-5


def test_embed_images_records_failures_for_unreadable_images(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    monkeypatch.setattr(ei, "IMAGES_DIR", images_dir)
    monkeypatch.setattr(ei, "METADATA_PATH", images_dir / "metadata.json")
    monkeypatch.setattr(ei, "EMBEDDINGS_PATH", tmp_path / "embeddings" / "image_embeddings.json")
    monkeypatch.setattr(ei, "REPORT_PATH", tmp_path / "image_embedding_report.json")

    corrupt_path = images_dir / "bad.png"
    corrupt_path.write_bytes(b"not an image")

    metadata = [{"image_file": "bad.png", "source_pdf": "a.pdf", "page": 1}]
    (images_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(ei, "CLIPModel", type("M", (), {"from_pretrained": staticmethod(lambda name: FakeModel())}))
    monkeypatch.setattr(ei, "CLIPProcessor", type("P", (), {"from_pretrained": staticmethod(lambda name: FakeProcessor())}))

    ei.embed_images()

    report = json.loads((tmp_path / "image_embedding_report.json").read_text(encoding="utf-8"))
    assert report["images_failed"] == 1
    assert report["images_embedded"] == 0
