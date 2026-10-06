"""extraction_quality: the check on a document right after extraction.

Catches an extraction that went wrong, before the bad text is chunked, embedded and loaded. A document fails if:
  - MORE THAN 30% of its pages are nearly empty (fewer than 100 characters of extracted text). This also
    catches a scan whose OCR read nothing. The 100 was chosen by measuring the 12 papers: their sparsest page
    holds 229 characters and none of the 218 pages is under 200, so all of them pass with a wide margin;
  - ANY table could not be converted (a table the extraction lost is a hole in the document);
  - it was read by OCR and the text is unreadable: fewer than 60% of its words are real English words. Measured: the 12
    papers' own text scores 0.89 to 0.90, a clear handout read well by Tesseract 0.96, the same handout read badly 0.46.
    Plenty of characters is not enough: gibberish would be chunked, embedded and answered from.
A failed document is not ingested: it goes on the review list with the reason and these numbers.
"""
import statistics
from dataclasses import asdict, dataclass, field

MIN_READABLE_SHARE = 0.60       # an OCR'd document with a smaller share of real English words is unreadable
MIN_WORDS_TO_JUDGE = 30         # fewer words than this: too little text to judge, the empty-page rule decides
NEARLY_EMPTY_CHARS = 100        # a page with fewer characters than this is nearly empty
MAX_NEARLY_EMPTY_SHARE = 0.30   # more than this share of nearly empty pages fails the document

TOO_MANY_EMPTY_PAGES = "too_many_empty_pages"
TABLE_CONVERSION_FAILED = "table_conversion_failed"
NO_PAGES = "no_pages"
OCR_UNREADABLE = "ocr_unreadable"

_vocabulary = None


def english_vocabulary() -> frozenset:
    """The words of spaCy's small English model (already installed for the personal-data check). Empty when it is not
    available: the readability check is then skipped, never a reason to reject a document."""
    global _vocabulary
    if _vocabulary is None:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "attribute_ruler", "lemmatizer"])
            _vocabulary = frozenset(s.lower() for s in nlp.vocab.strings if s.isalpha() and len(s) >= 2)
        except Exception:
            _vocabulary = frozenset()
    return _vocabulary


def readable_share(text: str, vocabulary=None):
    """(share of the words of 3 letters or more that are real English words, number of words), or (None, words) when
    there are too few words or no vocabulary to judge."""
    import re

    vocabulary = english_vocabulary() if vocabulary is None else vocabulary
    words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", text or "")]
    if not vocabulary or len(words) < MIN_WORDS_TO_JUDGE:
        return None, len(words)
    return sum(w in vocabulary for w in words) / len(words), len(words)


@dataclass
class QualityResult:
    passed: bool
    reason_code: str = None
    reason: str = None
    reason_codes: list = field(default_factory=list)  # every rule that failed (reason_code is the first)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def check_extraction(result, vocabulary=None) -> QualityResult:
    """Judges an extraction.ExtractionResult. `vocabulary` is for tests (the real one is spaCy's)."""
    page_chars = list(result.page_chars)
    pages = len(page_chars)
    empty_pages = [number for number, chars in enumerate(page_chars, 1) if chars < NEARLY_EMPTY_CHARS]
    share = len(empty_pages) / pages if pages else 1.0
    metrics = {
        "pages": pages,
        "blocks": len(result.blocks),
        "chars_total": sum(page_chars),
        "chars_median_per_page": statistics.median(page_chars) if page_chars else 0,
        "nearly_empty_pages": empty_pages,
        "nearly_empty_share": round(share, 3),
        "tables_found": result.tables_found,
        "tables_failed": len(result.tables_failed),
        "ocr_used": result.ocr_used,
        "seconds": result.seconds,
    }
    share_of_words = None
    if result.ocr_used:
        share_of_words, word_count = readable_share(" ".join(b["text"] for b in result.blocks), vocabulary)
        metrics["readable_share"] = None if share_of_words is None else round(share_of_words, 3)
        metrics["words"] = word_count

    codes, reasons = [], []
    if pages == 0:
        codes.append(NO_PAGES)
        reasons.append("The extraction found no pages.")
    elif share > MAX_NEARLY_EMPTY_SHARE:
        codes.append(TOO_MANY_EMPTY_PAGES)
        reasons.append(
            f"{len(empty_pages)} of {pages} pages ({share:.0%}) have fewer than {NEARLY_EMPTY_CHARS} characters of text "
            f"(the limit is {MAX_NEARLY_EMPTY_SHARE:.0%}): pages {_page_list(empty_pages)}."
        )
    if share_of_words is not None and share_of_words < MIN_READABLE_SHARE:
        codes.append(OCR_UNREADABLE)
        reasons.append(
            f"The OCR text looks unreadable: only {share_of_words:.0%} of its words are real English words "
            f"(the limit is {MIN_READABLE_SHARE:.0%}). The image or scan is probably too small or too blurred; "
            "upload a sharper one (about 1,500 pixels wide or more, or a PDF)."
        )
    if result.tables_failed:
        codes.append(TABLE_CONVERSION_FAILED)
        failed_pages = sorted({t["page"] for t in result.tables_failed if t.get("page")})
        reasons.append(
            f"{len(result.tables_failed)} table(s) could not be converted"
            + (f" (pages {_page_list(failed_pages)})." if failed_pages else ".")
        )

    if not codes:
        return QualityResult(passed=True, metrics=metrics)
    return QualityResult(passed=False, reason_code=codes[0], reason=" ".join(reasons), reason_codes=codes, metrics=metrics)


def _page_list(numbers: list, limit: int = 15) -> str:
    shown = ", ".join(str(n) for n in numbers[:limit])
    return shown + (f" and {len(numbers) - limit} more" if len(numbers) > limit else "")
