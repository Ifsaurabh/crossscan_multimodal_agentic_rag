"""intake_check: the first step for every uploaded document.

Runs on one file, before any extraction, and answers three questions:
  1. Is it something we accept? Only PDF files and image files (JPEG, PNG, TIFF, BMP, WEBP) are, judged
     by what the file CONTAINS, not by its extension. It must open, must not be password-protected,
     and must not be empty or blank. There is no limit on file size or page count.
  2. Has it changed? The SHA-256 of the content is compared with the hash of the version already
     ingested (`active_hash`, from the manifest): no earlier version = new, a different hash =
     changed, the same hash = unchanged (skipped without opening the file).
  3. What kind of pages does it have? Every page is labelled "text" (any extractable text at all),
     "image_only" (no text, but there is an image or drawing, so it needs OCR) or "blank". The
     document is then "text", "scanned" (every page image-only) or "mixed". An image file is a scanned
     document; a multi-page TIFF is handled page by page.

A file that is rejected is flagged for review and goes no further (the worker moves it to failed/).
The check needs no model and no database: the caller looks up `active_hash` itself.

Try it on a file (from the project root):  PYTHONPATH=src python -m ingestion.intake_check path/to/file.pdf
"""
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
from PIL import Image, ImageSequence, UnidentifiedImageError

ACCEPTED_IMAGE_FORMATS = ("JPEG", "PNG", "TIFF", "BMP", "WEBP")
ACCEPTED_DESCRIPTION = "PDF files and image files (JPEG, PNG, TIFF, BMP, WEBP)"

HASH_CHUNK_BYTES = 1024 * 1024
HEADER_BYTES = 1024  # a PDF reader accepts the "%PDF-" marker anywhere in the first kilobyte

# What happened to the file.
ACCEPT = "accept"
SKIP_UNCHANGED = "skip_unchanged"
REJECT = "reject"

# Why it was rejected or skipped.
EMPTY_FILE = "empty_file"
UNREADABLE = "unreadable"
UNSUPPORTED_TYPE = "unsupported_type"
CORRUPT = "corrupt"
PASSWORD_PROTECTED = "password_protected"
BLANK = "blank"
UNCHANGED = "unchanged"

# How a page is labelled.
TEXT = "text"
IMAGE_ONLY = "image_only"
BLANK_PAGE = "blank"

# How the whole document is labelled.
DOC_TEXT = "text"
DOC_SCANNED = "scanned"
DOC_MIXED = "mixed"

# Version of the file compared with what is already ingested.
NEW = "new"
CHANGED = "changed"


@dataclass
class IntakeResult:
    file_name: str
    outcome: str = REJECT
    reason_code: str = None
    reason: str = None
    size_bytes: int = 0
    content_hash: str = None
    version_status: str = None   # "new", "changed" or "unchanged"
    file_type: str = None        # "pdf" or "image"
    detected_format: str = None  # "PDF", "JPEG", "PNG", ...
    pages: int = 0
    text_pages: int = 0
    image_only_pages: int = 0
    blank_pages: int = 0
    page_labels: list = field(default_factory=list)
    document_label: str = None   # "text", "scanned" or "mixed"

    @property
    def accepted(self) -> bool:
        return self.outcome == ACCEPT

    def to_dict(self) -> dict:
        return asdict(self)


def file_hash(path) -> str:
    """SHA-256 of the file content, read in chunks so a very large file never has to fit in memory.
    Same value as ingestion_manifest.compute_content_hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject(result: IntakeResult, code: str, reason: str) -> IntakeResult:
    result.outcome, result.reason_code, result.reason = REJECT, code, reason
    return result


def _unsupported_message(path: Path, detected: str = None) -> str:
    extension = path.suffix.lower()
    what = f"a {detected} image" if detected else (f"a {extension} file" if extension else "this file")
    message = f"Unsupported file type: {what}. Only {ACCEPTED_DESCRIPTION} are accepted."
    if extension in (".heic", ".heif"):
        message += " HEIC photos must be converted to JPEG or PNG first."
    return message


def _document_label(page_labels: list):
    """text = every counted page has text, scanned = none has, mixed = some of each.
    Blank pages do not count either way. None when every page is blank."""
    counted = [label for label in page_labels if label != BLANK_PAGE]
    if not counted:
        return None
    if all(label == TEXT for label in counted):
        return DOC_TEXT
    if all(label == IMAGE_ONLY for label in counted):
        return DOC_SCANNED
    return DOC_MIXED


def _record_pages(result: IntakeResult, page_labels: list) -> IntakeResult:
    """Stores the page labels and counts, and rejects a document whose every page is blank."""
    result.page_labels = page_labels
    result.pages = len(page_labels)
    result.text_pages = page_labels.count(TEXT)
    result.image_only_pages = page_labels.count(IMAGE_ONLY)
    result.blank_pages = page_labels.count(BLANK_PAGE)
    result.document_label = _document_label(page_labels)
    if result.document_label is None:
        return _reject(result, BLANK, "Every page is blank: there is nothing to read.")
    result.outcome = ACCEPT
    return result


def _label_pdf_page(page) -> str:
    if page.get_text("text").strip():
        return TEXT  # any extractable text at all counts, there is no minimum
    # No text. An image or a drawing means there is something on the page that OCR must read.
    if page.get_images(full=True) or page.get_cdrawings():
        return IMAGE_ONLY
    return BLANK_PAGE


def _analyse_pdf(path: Path, result: IntakeResult) -> IntakeResult:
    result.file_type, result.detected_format = "pdf", "PDF"
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        return _reject(result, CORRUPT, f"The PDF could not be opened (it may be corrupt): {e}")
    try:
        if doc.needs_pass:
            return _reject(result, PASSWORD_PROTECTED, "The PDF is password-protected and cannot be read.")
        if doc.page_count == 0:
            return _reject(result, CORRUPT, "The PDF has no pages.")
        try:
            labels = [_label_pdf_page(page) for page in doc]
        except Exception as e:
            return _reject(result, CORRUPT, f"A page of the PDF could not be read (it may be corrupt): {e}")
    finally:
        doc.close()
    return _record_pages(result, labels)


def _is_blank_image(frame) -> bool:
    """True when every pixel has the same value (a plain white or black page)."""
    low, high = frame.convert("L").getextrema()
    return low == high


def _analyse_image(path: Path, result: IntakeResult) -> IntakeResult:
    try:
        image = Image.open(path)
    except UnidentifiedImageError:
        return _reject(result, UNSUPPORTED_TYPE, _unsupported_message(path))
    except Exception as e:
        return _reject(result, CORRUPT, f"The image could not be opened: {e}")
    try:
        result.file_type, result.detected_format = "image", image.format
        if image.format not in ACCEPTED_IMAGE_FORMATS:
            return _reject(result, UNSUPPORTED_TYPE, _unsupported_message(path, detected=image.format))
        try:
            labels = [BLANK_PAGE if _is_blank_image(frame) else IMAGE_ONLY for frame in ImageSequence.Iterator(image)]
        except Exception as e:
            return _reject(result, CORRUPT, f"The image could not be read (it may be corrupt or truncated): {e}")
    finally:
        image.close()
    return _record_pages(result, labels)


def check_document(path, source_name: str = None, active_hash: str = None) -> IntakeResult:
    """Runs the intake check on one local file.

    `source_name` is the document's name in the manifest (default: the file name).
    `active_hash` is the content hash of the version already ingested under that name, or None
    if there is none (ingestion_manifest.get_active_hash)."""
    path = Path(path)
    result = IntakeResult(file_name=source_name or path.name)

    try:
        result.size_bytes = path.stat().st_size
    except OSError as e:
        return _reject(result, UNREADABLE, f"The file could not be read: {e}")
    if result.size_bytes == 0:
        return _reject(result, EMPTY_FILE, "The file is empty.")

    try:
        result.content_hash = file_hash(path)
        with open(path, "rb") as f:
            header = f.read(HEADER_BYTES)
    except OSError as e:
        return _reject(result, UNREADABLE, f"The file could not be read: {e}")

    if active_hash is None:
        result.version_status = NEW
    elif active_hash == result.content_hash:
        result.version_status = UNCHANGED
        result.outcome, result.reason_code = SKIP_UNCHANGED, UNCHANGED
        result.reason = "Already ingested with exactly this content: nothing to do."
        return result  # not opened: it passed the checks when it was ingested
    else:
        result.version_status = CHANGED

    if b"%PDF-" in header:
        return _analyse_pdf(path, result)
    return _analyse_image(path, result)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: PYTHONPATH=src python -m ingestion.intake_check path/to/file")
    print(json.dumps(check_document(sys.argv[1]).to_dict(), indent=2))
