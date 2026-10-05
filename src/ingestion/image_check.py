"""image_check: proves that everything the ingestion worker needs is present and loads, with the network off.

It runs during the image build (Dockerfile.worker), in the pull-request build check, and can be run by hand:

    PYTHONPATH=src python -m ingestion.image_check            # Prompt Guard missing = a warning
    PYTHONPATH=src python -m ingestion.image_check --require-prompt-guard

Each check prints one line. The exit code is 1 if any required check failed. Tesseract is checked only with
`--tesseract` (the image has it; a laptop may not). Nothing here touches the database or the bucket.
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pymupdf

SAMPLE_TEXT = (
    "Lung nodule detection with convolutional networks. We describe a method for finding small nodules in "
    "chest CT scans and report its sensitivity on a public benchmark. Contact the authors at jane.doe@example.com."
)


def _sample_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 520, 400), SAMPLE_TEXT * 3, fontsize=11)
    doc.save(str(path))


def check_tesseract():
    from ingestion import extraction

    command = extraction.find_tesseract()
    if command is None:
        raise RuntimeError("Tesseract was not found (TESSERACT_CMD or the PATH)")
    languages = subprocess.run([command, "--list-langs"], capture_output=True, text=True, timeout=30).stdout
    if "eng" not in languages.split():
        raise RuntimeError(f"Tesseract has no English language data: {languages!r}")
    return f"{command}, languages: {' '.join(languages.split()[4:])}"


def check_tiktoken():
    from ingestion.chunking_config import TOKENIZER_ENCODING
    import tiktoken

    count = len(tiktoken.get_encoding(TOKENIZER_ENCODING).encode("a short sentence"))
    return f"{TOKENIZER_ENCODING}, {count} tokens"


def check_text_model():
    from ingestion import embed_text
    from shared.embedding_config import TEXT_MODEL_NAME

    vectors = embed_text.embed_texts(embed_text.load_text_model(), ["a short sentence"])
    if len(vectors[0]) != 768:
        raise RuntimeError(f"expected 768 dimensions, got {len(vectors[0])}")
    return f"{TEXT_MODEL_NAME}, {len(vectors[0])} dimensions"


def check_image_model():
    from PIL import Image

    from ingestion import embed_images
    from shared.embedding_config import IMAGE_MODEL_NAME

    model, processor = embed_images.load_image_model()
    vector = embed_images.embed_pil_image(model, processor, Image.new("RGB", (64, 64), "white"))
    if len(vector) != 512:
        raise RuntimeError(f"expected 512 dimensions, got {len(vector)}")
    return f"{IMAGE_MODEL_NAME}, {len(vector)} dimensions"


def check_redaction():
    from ingestion import ingestion_guardrails

    redacted, count = ingestion_guardrails.redact_pii("Write to jane.doe@example.com for the data.")
    if count < 1 or "jane.doe@example.com" in redacted:
        raise RuntimeError(f"an e-mail address was not redacted: {redacted!r}")
    return f"Presidio and spaCy, {count} redaction"


def check_prompt_guard():
    from shared import query_guardrail

    result = query_guardrail.classify_injection("The results are shown in Table 2.")
    if not result["available"]:
        raise RuntimeError("Prompt Guard could not be loaded (no model, or no licence/token at build time)")
    return f"{query_guardrail.PROMPT_GUARD_MODEL}, score {result['score']:.3f}"


def check_extraction():
    from ingestion import extraction, extraction_quality
    from ingestion.intake_check import DOC_TEXT, IntakeResult

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sample.pdf"
        _sample_pdf(path)
        intake = IntakeResult(file_name="check/sample.pdf", pages=1, text_pages=1, document_label=DOC_TEXT)
        result = extraction.extract_document(path, intake)
    if not result.blocks:
        raise RuntimeError("Docling returned no text for a one-page PDF")
    quality = extraction_quality.check_extraction(result)
    return f"Docling, {len(result.blocks)} blocks, {result.page_chars[0]} characters, quality {'ok' if quality.passed else 'FAILED'}"


def run(require_prompt_guard: bool = False, tesseract: bool = False) -> bool:
    checks = [("tiktoken", check_tiktoken, True), ("redaction", check_redaction, True),
              ("text model", check_text_model, True), ("image model", check_image_model, True),
              ("prompt guard", check_prompt_guard, require_prompt_guard), ("docling", check_extraction, True)]
    if tesseract:
        checks.insert(0, ("tesseract", check_tesseract, True))
    ok = True
    for name, check, required in checks:
        started = time.perf_counter()
        try:
            detail = check()
            print(f"OK       {name:<13} {detail} ({time.perf_counter() - started:.1f}s)", flush=True)
        except Exception as e:
            label = "FAILED  " if required else "WARNING "
            print(f"{label} {name:<13} {type(e).__name__}: {e}", flush=True)
            ok = ok and not required
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--require-prompt-guard", action="store_true", help="a missing Prompt Guard is a failure")
    parser.add_argument("--tesseract", action="store_true", help="also check that Tesseract and its English data are installed")
    args = parser.parse_args(argv)
    sys.exit(0 if run(args.require_prompt_guard, args.tesseract) else 1)


if __name__ == "__main__":
    main()
