"""extraction_quality: the check right after extraction - mostly empty pages, or a table that was lost."""
import json

from ingestion import extraction as ex
from ingestion import extraction_quality as eq


def extraction(page_chars, tables_failed=(), blocks=5, ocr=False):
    return ex.ExtractionResult(
        source_name="paper.pdf", blocks=[{}] * blocks, pages=len(page_chars), page_chars=list(page_chars),
        tables_found=len(tables_failed), tables_failed=list(tables_failed), ocr_used=ocr, seconds=1.5,
    )


def test_the_limits_are_the_agreed_ones():
    assert eq.NEARLY_EMPTY_CHARS == 100 and eq.MAX_NEARLY_EMPTY_SHARE == 0.30


def test_a_well_extracted_document_passes_and_reports_its_numbers():
    result = eq.check_extraction(extraction([3000, 4000, 2500, 500], blocks=40))

    assert result.passed and result.reason_code is None and result.reason is None and result.reason_codes == []
    assert result.metrics["pages"] == 4 and result.metrics["blocks"] == 40 and result.metrics["chars_total"] == 10000
    assert result.metrics["chars_median_per_page"] == 2750 and result.metrics["nearly_empty_pages"] == []
    assert result.metrics["nearly_empty_share"] == 0 and result.metrics["ocr_used"] is False and result.metrics["seconds"] == 1.5


def test_a_page_is_nearly_empty_below_100_characters_not_at_100():
    assert eq.check_extraction(extraction([100] * 10)).metrics["nearly_empty_pages"] == []
    assert eq.check_extraction(extraction([99] * 10)).metrics["nearly_empty_pages"] == list(range(1, 11))


def test_exactly_thirty_percent_nearly_empty_pages_still_passes():
    result = eq.check_extraction(extraction([0, 0, 0] + [2000] * 7))  # 3 of 10

    assert result.passed and result.metrics["nearly_empty_share"] == 0.3


def test_more_than_thirty_percent_nearly_empty_pages_fails_and_names_the_pages():
    result = eq.check_extraction(extraction([0, 50, 99, 20] + [2000] * 6))  # 4 of 10

    assert not result.passed and result.reason_code == eq.TOO_MANY_EMPTY_PAGES
    assert "4 of 10 pages (40%)" in result.reason and "fewer than 100 characters" in result.reason and "limit is 30%" in result.reason
    assert "pages 1, 2, 3, 4" in result.reason and result.metrics["nearly_empty_pages"] == [1, 2, 3, 4]


def test_a_scan_that_ocr_read_nothing_from_fails():
    result = eq.check_extraction(extraction([0] * 8, blocks=0, ocr=True))

    assert not result.passed and result.reason_code == eq.TOO_MANY_EMPTY_PAGES and result.metrics["nearly_empty_share"] == 1.0


def test_one_lost_table_fails_the_document_even_when_every_page_is_full():
    result = eq.check_extraction(extraction([3000, 3000], tables_failed=[{"page": 2, "error": "ValueError: x"}]))

    assert not result.passed and result.reason_code == eq.TABLE_CONVERSION_FAILED
    assert "1 table(s) could not be converted (pages 2)" in result.reason and result.metrics["tables_failed"] == 1


def test_a_lost_table_without_a_known_page_is_still_reported():
    result = eq.check_extraction(extraction([3000], tables_failed=[{"page": None, "error": "e"}]))

    assert result.reason == "1 table(s) could not be converted."


def test_both_problems_are_reported_with_the_page_rule_first():
    result = eq.check_extraction(extraction([0, 0, 0, 2000], tables_failed=[{"page": 4, "error": "e"}]))

    assert not result.passed and result.reason_code == eq.TOO_MANY_EMPTY_PAGES
    assert result.reason_codes == [eq.TOO_MANY_EMPTY_PAGES, eq.TABLE_CONVERSION_FAILED]
    assert "3 of 4 pages" in result.reason and "table(s) could not be converted" in result.reason


def test_a_document_with_no_pages_fails():
    result = eq.check_extraction(extraction([]))

    assert not result.passed and result.reason_code == eq.NO_PAGES and result.metrics["pages"] == 0


def test_a_long_list_of_empty_pages_is_shortened():
    result = eq.check_extraction(extraction([0] * 30))

    assert "pages 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 and 15 more" in result.reason


def test_the_result_can_be_stored_as_plain_data():
    data = eq.check_extraction(extraction([0, 0, 0, 2000], tables_failed=[{"page": 1, "error": "e"}])).to_dict()

    assert json.loads(json.dumps(data)) == data
    assert set(data) == {"passed", "reason_code", "reason", "reason_codes", "metrics"}


# ---------- OCR text that is not readable ----------

VOCABULARY = frozenset("the lung cancer can be detected with low dose screening scan during you will lie down in a donut like "
                       "structure while rays are passed through your body computers then use these to produce images of inside".split())
READABLE = ("the lung cancer can be detected with a low dose screening scan during the scan you will lie down in a donut like "
            "structure while rays are passed through your body computers then use these rays to produce images of the inside")
GIBBERISH = ("Lang cancer can tmetinebe detec wih nde ceing CT outed nga sn Dantes youl ie dors ina doo ta aya pe Bota keer "
             "Caner toh SSCESGKE GSLs toes ye drze die hom te creer manypeonle foe kre ccrcal acl Igcanethechon Einitentalsomtner")


def ocr_extraction(text, ocr=True):
    blocks = [{"label": "text", "page": 1, "text": text}]
    return ex.ExtractionResult(source_name="page.jpg", blocks=blocks, pages=1, page_chars=[len(text)], ocr_used=ocr, seconds=3.0)


def test_readable_ocr_text_passes_and_reports_its_share():
    result = eq.check_extraction(ocr_extraction(READABLE), vocabulary=VOCABULARY)

    assert result.passed and result.metrics["readable_share"] == 1.0 and result.metrics["words"] >= eq.MIN_WORDS_TO_JUDGE


def test_unreadable_ocr_text_is_rejected_even_though_it_has_plenty_of_characters():
    result = eq.check_extraction(ocr_extraction(GIBBERISH), vocabulary=VOCABULARY)

    assert not result.passed and result.reason_code == eq.OCR_UNREADABLE == "ocr_unreadable"
    assert result.metrics["nearly_empty_pages"] == [] and result.metrics["readable_share"] < eq.MIN_READABLE_SHARE
    assert "unreadable" in result.reason and "60%" in result.reason and "sharper" in result.reason


def test_text_that_was_not_read_by_ocr_is_never_judged_for_readability():
    result = eq.check_extraction(ocr_extraction(GIBBERISH, ocr=False), vocabulary=VOCABULARY)

    assert result.passed and "readable_share" not in result.metrics


def test_too_few_words_are_left_to_the_empty_page_rule():
    result = eq.check_extraction(ocr_extraction("xqzv wklm " * 8 + "a" * 120), vocabulary=VOCABULARY)

    assert result.passed and result.metrics["readable_share"] is None


def test_without_a_vocabulary_the_check_is_skipped_never_a_reason_to_reject():
    share, words = eq.readable_share(GIBBERISH, vocabulary=frozenset())

    assert share is None and words > 0
    assert eq.check_extraction(ocr_extraction(GIBBERISH), vocabulary=frozenset()).passed


def test_the_limits_for_readability_are_the_measured_ones():
    assert eq.MIN_READABLE_SHARE == 0.60 and eq.MIN_WORDS_TO_JUDGE == 30


def test_the_real_english_vocabulary_separates_a_clear_paragraph_from_gibberish():
    clear, _ = eq.readable_share(READABLE + " " + READABLE)
    garbled, _ = eq.readable_share(GIBBERISH)

    if clear is None:
        return  # spaCy's English model is not installed here: the check is skipped, as designed
    assert clear > 0.9 and garbled < eq.MIN_READABLE_SHARE
