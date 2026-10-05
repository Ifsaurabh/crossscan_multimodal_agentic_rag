"""extraction_quality: the check on a document right after extraction.

Catches an extraction that went wrong, before the bad text is chunked, embedded and loaded. A document fails if:
  - MORE THAN 30% of its pages are nearly empty (fewer than 100 characters of extracted text). This also
    catches a scan whose OCR read nothing. The 100 was chosen by measuring the 12 papers: their sparsest page
    holds 229 characters and none of the 218 pages is under 200, so all of them pass with a wide margin;
  - ANY table could not be converted (a table the extraction lost is a hole in the document).
A failed document is not ingested: it goes on the review list with the reason and these numbers.
"""
import statistics
from dataclasses import asdict, dataclass, field

NEARLY_EMPTY_CHARS = 100        # a page with fewer characters than this is nearly empty
MAX_NEARLY_EMPTY_SHARE = 0.30   # more than this share of nearly empty pages fails the document

TOO_MANY_EMPTY_PAGES = "too_many_empty_pages"
TABLE_CONVERSION_FAILED = "table_conversion_failed"
NO_PAGES = "no_pages"


@dataclass
class QualityResult:
    passed: bool
    reason_code: str = None
    reason: str = None
    reason_codes: list = field(default_factory=list)  # every rule that failed (reason_code is the first)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def check_extraction(result) -> QualityResult:
    """Judges an extraction.ExtractionResult."""
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
