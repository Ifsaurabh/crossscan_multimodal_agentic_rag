import json

import pymupdf
from PIL import Image

import quality_check as qc


def make_pdf(path, num_pages=1):
    doc = pymupdf.open()
    for _ in range(num_pages):
        doc.new_page()
    doc.save(str(path))
    doc.close()


def test_check_text_files_passes_healthy_document(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    text_dir = tmp_path / "text"
    raw_dir.mkdir()
    text_dir.mkdir()

    monkeypatch.setattr(qc, "RAW_DIR", raw_dir)
    monkeypatch.setattr(qc, "TEXT_DIR", text_dir)

    make_pdf(raw_dir / "sample.pdf", num_pages=1)

    blocks = [
        {"label": "section_header", "text": "Introduction", "page": 1},
        {"label": "text", "text": "A" * 300, "page": 1},
    ]
    (text_dir / "sample.json").write_text(
        json.dumps({"source_pdf": "sample.pdf", "blocks": blocks}), encoding="utf-8"
    )

    results = qc.check_text_files()

    assert len(results) == 1
    assert results[0]["issues"] == []


def test_check_text_files_flags_missing_file(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    text_dir = tmp_path / "text"
    raw_dir.mkdir()
    text_dir.mkdir()

    monkeypatch.setattr(qc, "RAW_DIR", raw_dir)
    monkeypatch.setattr(qc, "TEXT_DIR", text_dir)

    make_pdf(raw_dir / "missing.pdf", num_pages=1)

    results = qc.check_text_files()

    assert "missing_text_file" in results[0]["issues"]


def test_check_text_files_flags_low_chars_and_no_headings(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    text_dir = tmp_path / "text"
    raw_dir.mkdir()
    text_dir.mkdir()

    monkeypatch.setattr(qc, "RAW_DIR", raw_dir)
    monkeypatch.setattr(qc, "TEXT_DIR", text_dir)

    make_pdf(raw_dir / "thin.pdf", num_pages=1)

    blocks = [{"label": "text", "text": "short", "page": 1}]
    (text_dir / "thin.json").write_text(
        json.dumps({"source_pdf": "thin.pdf", "blocks": blocks}), encoding="utf-8"
    )

    results = qc.check_text_files()

    assert "low_chars_per_page" in results[0]["issues"]
    assert "no_headings_found" in results[0]["issues"]


def test_check_images_flags_and_removes_tiny_image(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    monkeypatch.setattr(qc, "IMAGES_DIR", images_dir)

    tiny_path = images_dir / "tiny.png"
    Image.new("RGB", (10, 10)).save(tiny_path)

    ok_path = images_dir / "ok.png"
    Image.new("RGB", (200, 200)).save(ok_path)

    metadata = [
        {"image_file": "tiny.png", "source_pdf": "a.pdf", "page": 1},
        {"image_file": "ok.png", "source_pdf": "a.pdf", "page": 1},
    ]
    (images_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    flagged = qc.check_images()

    assert len(flagged) == 1
    assert flagged[0]["file"] == "tiny.png"
    assert not tiny_path.exists()
    assert ok_path.exists()

    remaining = json.loads((images_dir / "metadata.json").read_text(encoding="utf-8"))
    assert len(remaining) == 1
    assert remaining[0]["image_file"] == "ok.png"
