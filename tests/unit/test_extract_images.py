import json

import pymupdf
from PIL import Image

import extract_images as ei


def make_pdf_with_image(pdf_path, img_path):
    Image.new("RGB", (100, 100), color="red").save(img_path)

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(0, 0, 100, 100), filename=str(img_path))
    doc.save(str(pdf_path))
    doc.close()


def make_pdf_without_image(pdf_path):
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()


def test_extract_images_extracts_embedded_image_and_metadata(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    images_dir = tmp_path / "images"
    raw_dir.mkdir()

    monkeypatch.setattr(ei, "RAW_DIR", raw_dir)
    monkeypatch.setattr(ei, "IMAGES_DIR", images_dir)

    make_pdf_with_image(raw_dir / "with_image.pdf", tmp_path / "source.png")

    ei.extract_images()

    metadata = json.loads((images_dir / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata) == 1
    assert metadata[0]["source_pdf"] == "with_image.pdf"
    assert metadata[0]["page"] == 1

    extracted_file = images_dir / metadata[0]["image_file"]
    assert extracted_file.exists()


def test_extract_images_handles_pdf_with_no_images(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    images_dir = tmp_path / "images"
    raw_dir.mkdir()

    monkeypatch.setattr(ei, "RAW_DIR", raw_dir)
    monkeypatch.setattr(ei, "IMAGES_DIR", images_dir)

    make_pdf_without_image(raw_dir / "no_image.pdf")

    ei.extract_images()

    metadata = json.loads((images_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == []
