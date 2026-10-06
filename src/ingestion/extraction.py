"""extraction: one document in, its text blocks out.

Docling reads the document into blocks (title, headings, paragraphs, lists, captions, tables ...), each with
its page number. The intake check (intake_check.py) already labelled every page, and that decides how:
  - a text document: no OCR at all (faster, and no noise from reading figures as if they were text);
  - a scanned or mixed PDF: OCR with Tesseract on, English only, with the page image enlarged 3 times
    (Docling's own default scale, which made the difference on low-resolution scans);
  - an image file (a photographed or screenshot page): Docling is NOT used, because it treats a whole-page image as one
    picture and its region-by-region OCR read a clear handout as gibberish (readable-word share 0.46). The image is
    enlarged to about 2,000 pixels wide and Tesseract reads the whole page at once (share 0.96 on the same file).
Tables are exported as markdown. A table that cannot be converted is not dropped silently: it is counted and
listed, so the quality check (extraction_quality.py) can stop the document.

Tesseract is found from the TESSERACT_CMD environment variable, then the PATH, then the usual per-user
Windows install. If OCR is needed and Tesseract is missing, OcrUnavailable is raised (a problem with the
machine, not with the document, so the message is retried).
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

TEXT_LABELS = {"text", "section_header", "footnote", "list_item", "caption", "title", "table", "formula"}

OCR_LANGUAGES = ["eng"]   # Docling's default is English, Spanish, French and German; only English is installed
OCR_SCALE = 3.0           # the page image is enlarged this many times before OCR
NEEDS_OCR_LABELS = ("scanned", "mixed")
IMAGE_OCR_TARGET_WIDTH = 2000   # an image file narrower than this is enlarged before OCR (Tesseract reads small text badly)
IMAGE_OCR_MAX_FACTOR = 4        # but never more than this many times
TESSERACT_TIMEOUT_SECONDS = 300


class OcrUnavailable(Exception):
    """OCR is needed for this document but Tesseract was not found on this machine."""


class ExtractionError(Exception):
    """Docling could not convert the document."""


@dataclass
class ExtractionResult:
    source_name: str
    blocks: list = field(default_factory=list)       # {"label", "level", "page", "text"}, in reading order
    pages: int = 0
    page_chars: list = field(default_factory=list)   # characters of extracted text on each page (page 1 first)
    tables_found: int = 0
    tables_failed: list = field(default_factory=list)  # {"page", "error"} for each table that could not be converted
    ocr_used: bool = False
    seconds: float = 0.0


def find_tesseract(environ=None, which=None, local_app_data=None):
    """The Tesseract command to use, or None. TESSERACT_CMD, then the PATH, then the per-user Windows install."""
    environ = os.environ if environ is None else environ
    which = shutil.which if which is None else which
    configured = environ.get("TESSERACT_CMD")
    if configured and (which(configured) or Path(configured).exists()):
        return configured
    on_path = which("tesseract")
    if on_path:
        return on_path
    base = local_app_data if local_app_data is not None else environ.get("LOCALAPPDATA")
    if base:
        windows_install = Path(base) / "Programs" / "Tesseract-OCR" / "tesseract.exe"
        if windows_install.exists():
            return str(windows_install)
    return None


def _ensure_tessdata(tesseract_cmd: str, environ=None) -> None:
    """A per-user Windows install keeps its language data next to the program: point Tesseract at it."""
    environ = os.environ if environ is None else environ
    if environ.get("TESSDATA_PREFIX"):
        return
    tessdata = Path(tesseract_cmd).parent / "tessdata"
    if tessdata.is_dir():
        environ["TESSDATA_PREFIX"] = str(tessdata)


def enlarged_for_ocr(frame):
    """A greyscale copy of one image, enlarged (smoothly) so that it is about IMAGE_OCR_TARGET_WIDTH pixels wide."""
    from PIL import Image

    grey = frame.convert("L")
    factor = min(IMAGE_OCR_MAX_FACTOR, max(1, -(-IMAGE_OCR_TARGET_WIDTH // max(1, grey.width))))
    if factor > 1:
        grey = grey.resize((grey.width * factor, grey.height * factor), Image.LANCZOS)
    return grey


def paragraphs_of(text: str) -> list:
    """Tesseract's output as paragraphs: blank lines separate them, a word split across two lines is joined again."""
    text = re.sub(r"-\n(?=[a-z])", "", text or "")
    paragraphs = (" ".join(part.split()) for part in re.split(r"\n\s*\n", text))
    return [p for p in paragraphs if len(p) >= 2]


def ocr_image_blocks(path, tesseract_cmd: str, run=subprocess.run):
    """(blocks, pages) of an image file, read by Tesseract on the whole page of each frame (a TIFF may have several)."""
    from PIL import Image, ImageSequence

    _ensure_tessdata(tesseract_cmd)
    blocks, pages = [], 0
    with Image.open(path) as image, tempfile.TemporaryDirectory() as folder:
        for number, frame in enumerate(ImageSequence.Iterator(image), 1):
            pages = number
            png = Path(folder) / f"page{number}.png"
            enlarged_for_ocr(frame).save(png, dpi=(300, 300))
            done = run([tesseract_cmd, str(png), "stdout", "-l", "+".join(OCR_LANGUAGES), "--psm", "3"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=TESSERACT_TIMEOUT_SECONDS)
            if done.returncode != 0:
                raise ExtractionError(f"Tesseract failed on page {number}: {(done.stderr or '').strip()[:200]}")
            blocks.extend({"label": "text", "level": None, "page": number, "text": p} for p in paragraphs_of(done.stdout))
    return blocks, pages


def build_converter(ocr: bool, tesseract_cmd: str = None) -> DocumentConverter:
    """A Docling converter set up explicitly, never left on its automatic choices."""
    options = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    if ocr:
        _ensure_tessdata(tesseract_cmd)
        options.ocr_options = TesseractCliOcrOptions(
            lang=OCR_LANGUAGES, tesseract_cmd=tesseract_cmd, scale=OCR_SCALE,
        )
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
    })


_converters = {}


def get_converter(ocr: bool, tesseract_cmd: str = None) -> DocumentConverter:
    """One converter per setting, kept for the life of the process: building one loads Docling's models."""
    key = (ocr, tesseract_cmd)
    if key not in _converters:
        _converters[key] = build_converter(ocr, tesseract_cmd)
    return _converters[key]


def blocks_from_document(document):
    """(blocks, tables_found, tables_failed) from a Docling document, in reading order."""
    blocks, tables_found, tables_failed = [], 0, []
    for item, _ in document.iterate_items():
        label = getattr(item, "label", None)
        prov = getattr(item, "prov", None)
        page = prov[0].page_no if prov else None
        caption = ""

        if label == "table":
            tables_found += 1
            try:
                text = item.export_to_markdown(doc=document)
            except Exception as e:
                tables_failed.append({"page": page, "error": f"{type(e).__name__}: {e}"})
                continue
            try:
                caption = (item.caption_text(document) or "").strip()  # "Table 3: precision of each model"
            except Exception:
                caption = ""  # a table without a readable caption is still a table
        else:
            text = getattr(item, "text", None)

        if label not in TEXT_LABELS or not text:
            continue
        level = getattr(item, "level", None) if label == "section_header" else None
        block = {"label": str(label), "level": level, "page": page, "text": text}
        if label == "table":
            block["caption"] = caption
        blocks.append(block)
    return blocks, tables_found, tables_failed


def needs_ocr(intake_result) -> bool:
    """OCR is used when the intake check found image-only pages (a scan, a mixed document, any image file)."""
    return intake_result.document_label in NEEDS_OCR_LABELS


def extract_document(path, intake_result, get_converter_fn=get_converter, find_tesseract_fn=find_tesseract,
                     ocr_image_fn=ocr_image_blocks) -> ExtractionResult:
    """Extracts one document that passed the intake check. `intake_result` is its intake_check.IntakeResult."""
    ocr = needs_ocr(intake_result)
    tesseract_cmd = None
    if ocr:
        tesseract_cmd = find_tesseract_fn()
        if tesseract_cmd is None:
            raise OcrUnavailable(
                f"{intake_result.file_name} has {intake_result.image_only_pages} image-only page(s) and needs OCR, "
                "but Tesseract was not found (set TESSERACT_CMD or put it on the PATH)."
            )

    started = time.perf_counter()
    if ocr and getattr(intake_result, "file_type", None) == "image":
        # An image file: Docling would see one picture and lose the text (see the module notes), so Tesseract reads it directly.
        try:
            blocks, frames = ocr_image_fn(path, tesseract_cmd)
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"The image {intake_result.file_name} could not be read: {type(e).__name__}: {e}") from e
        tables_found, tables_failed = 0, []
        pages = intake_result.pages or frames
    else:
        try:
            document = get_converter_fn(ocr, tesseract_cmd).convert(str(path)).document
        except Exception as e:
            raise ExtractionError(f"Docling could not convert {intake_result.file_name}: {type(e).__name__}: {e}") from e

        blocks, tables_found, tables_failed = blocks_from_document(document)
        pages = intake_result.pages or len(getattr(document, "pages", {}) or {})
    page_chars = [0] * pages
    for block in blocks:
        if block["page"] and 1 <= block["page"] <= pages:
            page_chars[block["page"] - 1] += len(block["text"])

    return ExtractionResult(
        source_name=intake_result.file_name, blocks=blocks, pages=pages, page_chars=page_chars,
        tables_found=tables_found, tables_failed=tables_failed, ocr_used=ocr,
        seconds=round(time.perf_counter() - started, 2),
    )
