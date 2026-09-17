import json

from PIL import Image

import image_guardrails as igr


def fake_pipeline_factory(nsfw_scores_by_filename):
    """Returns a fake classifier function that scores based on the image's filename,
    avoiding loading the real (heavy) NSFW model in unit tests."""

    def fake_classifier(img):
        # The real image_guardrails.py passes a PIL Image with no filename attached
        # after re-opening, so tests key off pixel color instead: pure red = "unsafe".
        pixel = img.getpixel((0, 0))
        is_flagged = pixel == (255, 0, 0)
        score = 0.9 if is_flagged else 0.01
        return [{"label": "nsfw", "score": score}, {"label": "normal", "score": 1 - score}]

    return fake_classifier


def test_run_image_guardrails_drops_flagged_and_keeps_safe(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    monkeypatch.setattr(igr, "IMAGES_DIR", images_dir)
    monkeypatch.setattr(igr, "METADATA_PATH", images_dir / "metadata.json")
    monkeypatch.setattr(igr, "REPORT_PATH", tmp_path / "image_guardrail_report.json")

    unsafe_path = images_dir / "unsafe.png"
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(unsafe_path)

    safe_path = images_dir / "safe.png"
    Image.new("RGB", (100, 100), color=(0, 255, 0)).save(safe_path)

    metadata = [
        {"image_file": "unsafe.png", "source_pdf": "a.pdf", "page": 1},
        {"image_file": "safe.png", "source_pdf": "a.pdf", "page": 2},
    ]
    (images_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(igr, "pipeline", lambda *a, **k: fake_pipeline_factory({}))

    igr.run_image_guardrails()

    assert not unsafe_path.exists()
    assert safe_path.exists()

    remaining = json.loads((images_dir / "metadata.json").read_text(encoding="utf-8"))
    assert len(remaining) == 1
    assert remaining[0]["image_file"] == "safe.png"

    report = json.loads((tmp_path / "image_guardrail_report.json").read_text(encoding="utf-8"))
    assert report["images_flagged"] == 1
    assert report["images_remaining"] == 1


def test_run_image_guardrails_handles_corrupt_image(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    monkeypatch.setattr(igr, "IMAGES_DIR", images_dir)
    monkeypatch.setattr(igr, "METADATA_PATH", images_dir / "metadata.json")
    monkeypatch.setattr(igr, "REPORT_PATH", tmp_path / "image_guardrail_report.json")

    corrupt_path = images_dir / "corrupt.png"
    corrupt_path.write_bytes(b"not a real image")

    metadata = [{"image_file": "corrupt.png", "source_pdf": "a.pdf", "page": 1}]
    (images_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(igr, "pipeline", lambda *a, **k: fake_pipeline_factory({}))

    igr.run_image_guardrails()

    remaining = json.loads((images_dir / "metadata.json").read_text(encoding="utf-8"))
    assert remaining == []
