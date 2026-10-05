"""intake_check: the first check on an uploaded file - what it is, whether it changed, what its pages hold.
Every file is generated here (PDFs with PyMuPDF, images with Pillow); nothing external is touched."""
import hashlib
import io
import json

import pymupdf
import pytest
from PIL import Image

from ingestion import ingestion_manifest
from ingestion import intake_check as ic


# ---------- generated files ----------

def noise_image(size=(240, 160)):
    """An image with real content (random noise), so it is never blank."""
    return Image.effect_noise(size, 80).convert("RGB")


def png_bytes():
    buffer = io.BytesIO()
    noise_image().save(buffer, format="PNG")
    return buffer.getvalue()


def save_pdf(doc, path, **kwargs):
    doc.save(path, **kwargs)
    doc.close()
    return path


def text_pdf(path, pages=1):
    doc = pymupdf.open()
    for number in range(pages):
        doc.new_page().insert_text((72, 72), f"Page {number + 1}: lung nodule detection results")
    return save_pdf(doc, path)


def image_pdf(path, pages=1):
    """A scanned-style PDF: every page is just a picture, with no text layer."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(page.rect, stream=png_bytes())
    return save_pdf(doc, path)


def mixed_pdf(path):
    """Page 1 has text, page 2 is a picture, page 3 is empty."""
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "A page with real text")
    page = doc.new_page()
    page.insert_image(page.rect, stream=png_bytes())
    doc.new_page()
    return save_pdf(doc, path)


def blank_pdf(path, pages=2):
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    return save_pdf(doc, path)


def drawing_pdf(path):
    """A page with a vector drawing and no text, no picture."""
    doc = pymupdf.open()
    doc.new_page().draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0, 0, 0))
    return save_pdf(doc, path)


def encrypted_pdf(path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    return save_pdf(doc, path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")


# ---------- the hash ----------

def test_the_hash_is_the_sha256_of_the_content_and_matches_the_manifests(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"some content" * 1000)

    assert ic.file_hash(path) == hashlib.sha256(b"some content" * 1000).hexdigest()
    assert ic.file_hash(path) == ingestion_manifest.compute_content_hash(path)


def test_a_file_larger_than_the_read_chunk_is_hashed_in_pieces_with_the_same_result(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "HASH_CHUNK_BYTES", 1000)
    data = bytes(range(256)) * 50  # 12,800 bytes: thirteen chunks
    path = tmp_path / "big.bin"
    path.write_bytes(data)

    assert ic.file_hash(path) == hashlib.sha256(data).hexdigest()


# ---------- a PDF with text ----------

def test_a_text_pdf_is_accepted_and_labelled_text(tmp_path):
    result = ic.check_document(text_pdf(tmp_path / "paper.pdf", pages=3))

    assert result.accepted and result.outcome == ic.ACCEPT and result.reason is None
    assert (result.file_type, result.detected_format) == ("pdf", "PDF")
    assert (result.pages, result.text_pages, result.image_only_pages, result.blank_pages) == (3, 3, 0, 0)
    assert result.page_labels == ["text", "text", "text"] and result.document_label == "text"
    assert result.version_status == ic.NEW and result.size_bytes > 0 and len(result.content_hash) == 64


def test_there_is_no_limit_on_the_number_of_pages(tmp_path):
    result = ic.check_document(text_pdf(tmp_path / "long.pdf", pages=60))

    assert result.accepted and result.pages == 60


# ---------- a scanned PDF ----------

def test_a_pdf_of_pictures_is_labelled_scanned(tmp_path):
    result = ic.check_document(image_pdf(tmp_path / "scan.pdf", pages=2))

    assert result.accepted and result.document_label == "scanned"
    assert (result.pages, result.text_pages, result.image_only_pages) == (2, 0, 2)
    assert result.page_labels == ["image_only", "image_only"]


def test_a_pdf_with_both_kinds_of_page_is_mixed_and_blank_pages_count_neither_way(tmp_path):
    result = ic.check_document(mixed_pdf(tmp_path / "mixed.pdf"))

    assert result.accepted and result.document_label == "mixed"
    assert result.page_labels == ["text", "image_only", "blank"]
    assert (result.text_pages, result.image_only_pages, result.blank_pages) == (1, 1, 1)


def test_blank_pages_do_not_stop_a_text_document_being_text(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "text")
    doc.new_page()
    result = ic.check_document(save_pdf(doc, tmp_path / "p.pdf"))

    assert result.document_label == "text" and result.blank_pages == 1


def test_a_page_with_only_a_vector_drawing_needs_ocr_so_it_is_image_only(tmp_path):
    result = ic.check_document(drawing_pdf(tmp_path / "drawing.pdf"))

    assert result.accepted and result.page_labels == ["image_only"] and result.document_label == "scanned"


def test_any_extractable_text_makes_a_page_a_text_page(tmp_path):
    """No minimum amount: a scan with a stray page number counts as text (the quality check after extraction catches that)."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=png_bytes())
    page.insert_text((72, 800), "7")
    result = ic.check_document(save_pdf(doc, tmp_path / "scan_with_number.pdf"))

    assert result.page_labels == ["text"] and result.document_label == "text"


# ---------- PDFs that are rejected ----------

def test_a_pdf_with_only_blank_pages_is_rejected(tmp_path):
    result = ic.check_document(blank_pdf(tmp_path / "blank.pdf"))

    assert not result.accepted and result.reason_code == ic.BLANK and "blank" in result.reason.lower()
    assert result.blank_pages == 2 and result.document_label is None


def test_a_password_protected_pdf_is_rejected(tmp_path):
    result = ic.check_document(encrypted_pdf(tmp_path / "locked.pdf"))

    assert not result.accepted and result.reason_code == ic.PASSWORD_PROTECTED
    assert "password" in result.reason.lower()


def test_a_corrupt_pdf_is_rejected(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not really a pdf at all")

    result = ic.check_document(path)

    assert not result.accepted and result.reason_code == ic.CORRUPT
    assert result.file_type == "pdf"


# ---------- images ----------

@pytest.mark.parametrize("fmt,extension", [
    ("PNG", ".png"), ("JPEG", ".jpg"), ("TIFF", ".tif"), ("BMP", ".bmp"), ("WEBP", ".webp"),
])
def test_every_accepted_image_format_is_a_scanned_document(tmp_path, fmt, extension):
    path = tmp_path / f"page{extension}"
    noise_image().save(path, format=fmt)

    result = ic.check_document(path)

    assert result.accepted and result.file_type == "image" and result.detected_format == fmt
    assert (result.pages, result.image_only_pages, result.text_pages) == (1, 1, 0)
    assert result.document_label == "scanned" and result.page_labels == ["image_only"]


def test_a_multi_page_tiff_is_handled_page_by_page(tmp_path):
    frames = [noise_image(), Image.new("RGB", (240, 160), "white"), noise_image()]
    path = tmp_path / "scan.tif"
    frames[0].save(path, format="TIFF", save_all=True, append_images=frames[1:])

    result = ic.check_document(path)

    assert result.accepted and result.pages == 3
    assert result.page_labels == ["image_only", "blank", "image_only"] and result.document_label == "scanned"


def test_a_plain_white_image_is_rejected_as_blank(tmp_path):
    path = tmp_path / "white.png"
    Image.new("RGB", (200, 100), "white").save(path)

    result = ic.check_document(path)

    assert not result.accepted and result.reason_code == ic.BLANK and result.blank_pages == 1


def test_a_truncated_image_is_rejected_as_corrupt(tmp_path):
    data = png_bytes()
    path = tmp_path / "cut.png"
    path.write_bytes(data[: len(data) // 2])

    result = ic.check_document(path)

    assert not result.accepted and result.reason_code == ic.CORRUPT and result.file_type == "image"


def test_an_image_format_that_is_not_accepted_is_rejected_and_named(tmp_path):
    path = tmp_path / "animation.gif"
    Image.new("P", (20, 20)).save(path, format="GIF")

    result = ic.check_document(path)

    assert not result.accepted and result.reason_code == ic.UNSUPPORTED_TYPE
    assert result.detected_format == "GIF" and "GIF" in result.reason and "JPEG, PNG, TIFF, BMP, WEBP" in result.reason


# ---------- what the file contains decides, not its name ----------

def test_a_pdf_with_an_image_extension_is_read_as_a_pdf(tmp_path):
    path = tmp_path / "scan.jpg"
    text_pdf(path)

    result = ic.check_document(path)

    assert result.accepted and result.file_type == "pdf"


def test_an_image_with_a_pdf_extension_is_read_as_an_image(tmp_path):
    path = tmp_path / "paper.pdf"
    noise_image().save(path, format="PNG")

    result = ic.check_document(path)

    assert result.accepted and result.file_type == "image" and result.detected_format == "PNG"


def test_a_pdf_marker_a_little_after_the_start_still_counts_as_a_pdf(tmp_path):
    source = text_pdf(tmp_path / "plain.pdf")
    path = tmp_path / "padded.pdf"
    path.write_bytes(b"junk before the header\n" + source.read_bytes())

    assert ic.check_document(path).file_type == "pdf"


# ---------- files that are not accepted at all ----------

def test_an_empty_file_is_rejected(tmp_path):
    path = tmp_path / "nothing.pdf"
    path.write_bytes(b"")

    result = ic.check_document(path)

    assert not result.accepted and result.reason_code == ic.EMPTY_FILE and result.size_bytes == 0


def test_a_missing_file_is_rejected_as_unreadable(tmp_path):
    result = ic.check_document(tmp_path / "gone.pdf")

    assert not result.accepted and result.reason_code == ic.UNREADABLE


@pytest.mark.parametrize("name,content", [
    ("notes.txt", b"just some text"),
    ("report.docx", b"PK\x03\x04 a zip container"),
    ("page.html", b"<html><body>hi</body></html>"),
    ("data.csv", b"a,b,c\n1,2,3\n"),
])
def test_other_document_types_are_rejected_with_the_accepted_ones_listed(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)

    result = ic.check_document(path)

    assert not result.accepted and result.reason_code == ic.UNSUPPORTED_TYPE
    assert result.file_type is None
    assert "PDF files and image files" in result.reason and path.suffix in result.reason


def test_a_heic_photo_is_rejected_with_a_hint_to_convert_it(tmp_path):
    path = tmp_path / "photo.heic"
    path.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)

    result = ic.check_document(path)

    assert not result.accepted and result.reason_code == ic.UNSUPPORTED_TYPE
    assert "HEIC" in result.reason and "converted" in result.reason


# ---------- new, changed, unchanged ----------

def test_a_document_with_no_earlier_version_is_new(tmp_path):
    result = ic.check_document(text_pdf(tmp_path / "a.pdf"), active_hash=None)

    assert result.version_status == "new" and result.accepted


def test_a_document_whose_content_differs_from_the_ingested_version_is_changed_and_still_accepted(tmp_path):
    result = ic.check_document(text_pdf(tmp_path / "a.pdf"), active_hash="0" * 64)

    assert result.version_status == "changed" and result.accepted and result.document_label == "text"


def test_an_unchanged_document_is_skipped_without_being_opened(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.7\nnot even a real pdf")  # would be rejected as corrupt if it were opened
    same = ic.file_hash(path)

    result = ic.check_document(path, active_hash=same)

    assert result.outcome == ic.SKIP_UNCHANGED and not result.accepted
    assert (result.version_status, result.reason_code) == ("unchanged", ic.UNCHANGED)
    assert result.pages == 0 and result.file_type is None  # never analysed


def test_the_name_in_the_result_defaults_to_the_file_name_and_can_be_overridden(tmp_path):
    path = text_pdf(tmp_path / "local-copy.pdf")

    assert ic.check_document(path).file_name == "local-copy.pdf"
    assert ic.check_document(path, source_name="paper.pdf").file_name == "paper.pdf"


# ---------- the result ----------

def test_the_result_is_plain_data_that_can_be_stored_or_logged(tmp_path):
    result = ic.check_document(mixed_pdf(tmp_path / "mixed.pdf"))

    data = result.to_dict()

    assert json.loads(json.dumps(data)) == data
    assert set(data) == {
        "file_name", "outcome", "reason_code", "reason", "size_bytes", "content_hash", "version_status", "file_type",
        "detected_format", "pages", "text_pages", "image_only_pages", "blank_pages", "page_labels", "document_label",
    }


def test_a_rejected_file_still_reports_its_hash_and_size_for_the_review_list(tmp_path):
    path = tmp_path / "locked.pdf"
    encrypted_pdf(path)

    result = ic.check_document(path)

    assert result.content_hash == ic.file_hash(path) and result.size_bytes == path.stat().st_size
