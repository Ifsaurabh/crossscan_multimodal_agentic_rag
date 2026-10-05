import pymupdf
from PIL import Image

from ingestion import extract_images as ei


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


def test_the_pictures_of_a_pdf_come_back_with_page_name_and_bytes(tmp_path):
    make_pdf_with_image(tmp_path / "with_image.pdf", tmp_path / "source.png")

    found = ei.extract_pdf_images(tmp_path / "with_image.pdf")

    assert len(found) == 1
    assert found[0]["page"] == 1 and found[0]["filename"] == "page1_img1." + found[0]["ext"]
    assert found[0]["bytes"] and isinstance(found[0]["bytes"], bytes)


def test_a_pdf_with_no_pictures_gives_an_empty_list(tmp_path):
    make_pdf_without_image(tmp_path / "no_image.pdf")
    assert ei.extract_pdf_images(tmp_path / "no_image.pdf") == []
