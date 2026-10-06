"""extraction: Docling set up for one document, and its output turned into blocks. Docling itself is faked."""
import json
from types import SimpleNamespace

import pytest

from ingestion import extraction as ex
from ingestion import intake_check as ic


# ---------- stand-ins for Docling ----------

class FakeItem:
    def __init__(self, label, text=None, page=1, level=None):
        self.label, self.text, self.level = label, text, level
        self.prov = [SimpleNamespace(page_no=page)] if page else []


class FakeTable(FakeItem):
    def __init__(self, page=1, markdown="| a | b |\n|---|---|\n| 1 | 2 |", error=None):
        super().__init__("table", None, page)
        self.markdown, self.error = markdown, error

    def export_to_markdown(self, doc=None):
        if self.error:
            raise self.error
        return self.markdown


class FakeCaptionedTable(FakeTable):
    def __init__(self, caption, **kwargs):
        super().__init__(**kwargs)
        self.caption = caption

    def caption_text(self, doc):
        if isinstance(self.caption, Exception):
            raise self.caption
        return self.caption


class FakeDocument:
    def __init__(self, items, pages=0):
        self.items = items
        self.pages = {n: None for n in range(1, pages + 1)}

    def iterate_items(self):
        return [(item, 0) for item in self.items]


class FakeConverter:
    def __init__(self, document=None, error=None):
        self.document, self.error, self.converted = document, error, []

    def convert(self, path):
        self.converted.append(path)
        if self.error:
            raise self.error
        return SimpleNamespace(document=self.document)


def intake(label="text", pages=3, image_only=0, name="paper.pdf"):
    return ic.IntakeResult(file_name=name, outcome=ic.ACCEPT, document_label=label, pages=pages, image_only_pages=image_only)


# ---------- blocks ----------

def test_blocks_keep_the_reading_order_the_page_and_the_heading_level():
    document = FakeDocument([
        FakeItem("title", "A Paper", page=1),
        FakeItem("section_header", "1. Introduction", page=1, level=1),
        FakeItem("text", "Some text.", page=1, level=7),
        FakeItem("list_item", "- a point", page=2),
    ])

    blocks, found, failed = ex.blocks_from_document(document)

    assert blocks == [
        {"label": "title", "level": None, "page": 1, "text": "A Paper"},
        {"label": "section_header", "level": 1, "page": 1, "text": "1. Introduction"},
        {"label": "text", "level": None, "page": 1, "text": "Some text."},  # only headings carry a level
        {"label": "list_item", "level": None, "page": 2, "text": "- a point"},
    ]
    assert (found, failed) == (0, [])


def test_labels_that_are_not_text_and_empty_text_are_left_out():
    document = FakeDocument([
        FakeItem("picture", "figure bytes"), FakeItem("page_header", "Journal 2025"), FakeItem("text", ""), FakeItem("text", None),
        FakeItem("text", "kept"),
    ])

    blocks, _, _ = ex.blocks_from_document(document)

    assert [b["text"] for b in blocks] == ["kept"]


def test_an_item_with_no_position_has_no_page():
    blocks, _, _ = ex.blocks_from_document(FakeDocument([FakeItem("text", "floating", page=None)]))

    assert blocks[0]["page"] is None


def test_a_table_becomes_a_markdown_block_and_is_counted():
    table = FakeTable(page=4, markdown="| a |\n|---|\n| 1 |")

    blocks, found, failed = ex.blocks_from_document(FakeDocument([table]))

    assert blocks == [{"label": "table", "level": None, "page": 4, "text": "| a |\n|---|\n| 1 |", "caption": ""}]
    assert (found, failed) == (1, [])


def test_a_table_that_cannot_be_converted_is_counted_and_listed_not_dropped_silently():
    broken = FakeTable(page=6, error=ValueError("merged cells"))

    blocks, found, failed = ex.blocks_from_document(FakeDocument([FakeItem("text", "before"), broken, FakeItem("text", "after")]))

    assert [b["text"] for b in blocks] == ["before", "after"]  # the bad table gives no block ...
    assert found == 1 and failed == [{"page": 6, "error": "ValueError: merged cells"}]  # ... but it is on record


# ---------- when OCR is used ----------

@pytest.mark.parametrize("label,expected", [("text", False), ("scanned", True), ("mixed", True), (None, False)])
def test_ocr_is_used_only_when_the_intake_check_found_image_only_pages(label, expected):
    assert ex.needs_ocr(intake(label)) is expected


# ---------- finding Tesseract ----------

def test_the_configured_command_wins_when_it_exists(tmp_path):
    exe = tmp_path / "tess.exe"
    exe.write_text("x")

    found = ex.find_tesseract(environ={"TESSERACT_CMD": str(exe)}, which=lambda name: "/usr/bin/tesseract")

    assert found == str(exe)


def test_without_a_valid_configured_command_the_path_is_searched():
    found = ex.find_tesseract(environ={"TESSERACT_CMD": "/nowhere/tess"}, which=lambda name: "/usr/bin/tesseract" if name == "tesseract" else None)

    assert found == "/usr/bin/tesseract"


def test_the_per_user_windows_install_is_the_last_resort(tmp_path):
    install = tmp_path / "Programs" / "Tesseract-OCR"
    install.mkdir(parents=True)
    (install / "tesseract.exe").write_text("x")

    found = ex.find_tesseract(environ={}, which=lambda name: None, local_app_data=str(tmp_path))

    assert found == str(install / "tesseract.exe")


def test_no_tesseract_anywhere_gives_none(tmp_path):
    assert ex.find_tesseract(environ={}, which=lambda name: None, local_app_data=str(tmp_path)) is None


def test_the_language_data_next_to_the_program_is_pointed_at_unless_already_set(tmp_path):
    (tmp_path / "tessdata").mkdir()
    cmd = str(tmp_path / "tesseract.exe")

    environ = {}
    ex._ensure_tessdata(cmd, environ)
    assert environ == {"TESSDATA_PREFIX": str(tmp_path / "tessdata")}

    kept = {"TESSDATA_PREFIX": "/already/set"}
    ex._ensure_tessdata(cmd, kept)
    assert kept == {"TESSDATA_PREFIX": "/already/set"}

    none = {}
    ex._ensure_tessdata(str(tmp_path / "other" / "tesseract"), none)
    assert none == {}


# ---------- the converter ----------

class RecordingConverter:
    def __init__(self, format_options=None):
        self.format_options = format_options


def test_a_text_document_gets_a_converter_with_ocr_off_and_tables_on(monkeypatch):
    monkeypatch.setattr(ex, "DocumentConverter", RecordingConverter)

    converter = ex.build_converter(ocr=False)

    for input_format in (ex.InputFormat.PDF, ex.InputFormat.IMAGE):
        options = converter.format_options[input_format].pipeline_options
        assert options.do_ocr is False and options.do_table_structure is True


def test_a_scanned_document_gets_tesseract_set_explicitly_in_english_at_three_times_scale(monkeypatch):
    monkeypatch.setattr(ex, "DocumentConverter", RecordingConverter)
    monkeypatch.setattr(ex, "_ensure_tessdata", lambda cmd, environ=None: None)

    converter = ex.build_converter(ocr=True, tesseract_cmd="/opt/tesseract")

    for input_format in (ex.InputFormat.PDF, ex.InputFormat.IMAGE):
        options = converter.format_options[input_format].pipeline_options
        ocr = options.ocr_options
        assert options.do_ocr is True and options.do_table_structure is True
        assert (ocr.lang, ocr.scale, ocr.tesseract_cmd) == (["eng"], 3.0, "/opt/tesseract")
        assert type(ocr).__name__ == "TesseractCliOcrOptions"  # never left on Docling's automatic engine choice


def test_a_converter_is_built_once_per_setting_and_reused(monkeypatch):
    built = []
    monkeypatch.setattr(ex, "_converters", {})
    monkeypatch.setattr(ex, "build_converter", lambda ocr, cmd=None: built.append((ocr, cmd)) or object())

    first = ex.get_converter(False)
    assert ex.get_converter(False) is first
    ex.get_converter(True, "/opt/tesseract")
    ex.get_converter(True, "/opt/tesseract")

    assert built == [(False, None), (True, "/opt/tesseract")]


# ---------- one document ----------

def converter_factory(converter, seen):
    def factory(ocr, cmd):
        seen.append((ocr, cmd))
        return converter
    return factory


def test_a_text_document_is_extracted_without_ocr_and_without_looking_for_tesseract(tmp_path):
    document = FakeDocument([
        FakeItem("section_header", "Intro", page=1, level=1), FakeItem("text", "x" * 300, page=1),
        FakeItem("text", "y" * 120, page=3), FakeTable(page=3, markdown="m" * 30),
    ])
    converter, seen = FakeConverter(document), []

    result = ex.extract_document(tmp_path / "paper.pdf", intake("text", pages=3), converter_factory(converter, seen),
                                 find_tesseract_fn=lambda: pytest.fail("no OCR, so no Tesseract needed"))

    assert seen == [(False, None)] and converter.converted == [str(tmp_path / "paper.pdf")]
    assert result.source_name == "paper.pdf" and not result.ocr_used and result.pages == 3
    assert [b["label"] for b in result.blocks] == ["section_header", "text", "text", "table"]
    assert result.page_chars == [len("Intro") + 300, 0, 120 + 30]  # characters per page, page 2 has none
    assert (result.tables_found, result.tables_failed) == (1, []) and result.seconds >= 0


@pytest.mark.parametrize("label", ["scanned", "mixed"])
def test_a_scanned_or_mixed_document_is_extracted_with_ocr_using_the_tesseract_found(tmp_path, label):
    converter, seen = FakeConverter(FakeDocument([FakeItem("text", "z" * 200, page=1)])), []

    result = ex.extract_document(tmp_path / "scan.pdf", intake(label, pages=2, image_only=2), converter_factory(converter, seen),
                                 find_tesseract_fn=lambda: "/usr/bin/tesseract")

    assert seen == [(True, "/usr/bin/tesseract")] and result.ocr_used is True and result.page_chars == [200, 0]


def test_an_image_file_is_read_by_tesseract_directly_and_docling_is_not_used(tmp_path):
    read = []
    image = ic.IntakeResult(file_name="page.jpg", outcome=ic.ACCEPT, file_type="image", document_label="scanned", pages=1, image_only_pages=1)

    def fake_ocr(path, tesseract_cmd):
        read.append((path, tesseract_cmd))
        return [{"label": "text", "level": None, "page": 1, "text": "Lung cancer can be detected early. " * 5}], 1

    result = ex.extract_document(tmp_path / "page.jpg", image, lambda ocr, cmd: pytest.fail("Docling must not be used for an image file"),
                                 find_tesseract_fn=lambda: "tess", ocr_image_fn=fake_ocr)

    assert read == [(tmp_path / "page.jpg", "tess")]
    assert result.ocr_used is True and result.pages == 1 and result.page_chars == [len("Lung cancer can be detected early. " * 5)]
    assert result.tables_found == 0 and result.tables_failed == []


def test_an_image_that_cannot_be_read_becomes_an_extraction_error_that_names_the_file(tmp_path):
    image = ic.IntakeResult(file_name="page.jpg", outcome=ic.ACCEPT, file_type="image", document_label="scanned", pages=1, image_only_pages=1)

    def broken(path, tesseract_cmd):
        raise OSError("disk full")

    with pytest.raises(ex.ExtractionError, match="page.jpg.*OSError.*disk full"):
        ex.extract_document(tmp_path / "page.jpg", image, lambda ocr, cmd: None, find_tesseract_fn=lambda: "tess", ocr_image_fn=broken)


def test_needing_ocr_without_tesseract_is_a_machine_problem_that_names_the_file(tmp_path):
    seen = []

    with pytest.raises(ex.OcrUnavailable, match="scan.pdf.*2 image-only.*Tesseract"):
        ex.extract_document(tmp_path / "scan.pdf", intake("scanned", pages=2, image_only=2, name="scan.pdf"),
                            converter_factory(FakeConverter(), seen), find_tesseract_fn=lambda: None)

    assert seen == []  # no converter was built


def test_a_docling_failure_becomes_an_extraction_error_that_keeps_the_cause(tmp_path):
    cause = RuntimeError("layout model crashed")

    with pytest.raises(ex.ExtractionError, match="paper.pdf.*RuntimeError.*layout model crashed") as failed:
        ex.extract_document(tmp_path / "paper.pdf", intake("text"), converter_factory(FakeConverter(error=cause), []))

    assert failed.value.__cause__ is cause


def test_failed_tables_are_carried_into_the_result(tmp_path):
    document = FakeDocument([FakeItem("text", "t" * 150, page=1), FakeTable(page=1, error=ValueError("bad"))])

    result = ex.extract_document(tmp_path / "p.pdf", intake("text", pages=1), converter_factory(FakeConverter(document), []))

    assert result.tables_found == 1 and result.tables_failed == [{"page": 1, "error": "ValueError: bad"}]


def test_the_page_count_comes_from_the_intake_check_or_else_from_docling(tmp_path):
    document = FakeDocument([FakeItem("text", "a" * 10, page=2)], pages=4)

    from_intake = ex.extract_document(tmp_path / "p.pdf", intake("text", pages=3), converter_factory(FakeConverter(document), []))
    fallback = ex.extract_document(tmp_path / "p.pdf", intake("text", pages=0), converter_factory(FakeConverter(document), []))

    assert from_intake.pages == 3 and fallback.pages == 4


def test_a_block_on_a_page_beyond_the_count_is_kept_but_not_counted_in_a_page(tmp_path):
    document = FakeDocument([FakeItem("text", "a" * 10, page=1), FakeItem("text", "b" * 10, page=9), FakeItem("text", "c" * 10, page=None)])

    result = ex.extract_document(tmp_path / "p.pdf", intake("text", pages=2), converter_factory(FakeConverter(document), []))

    assert len(result.blocks) == 3 and result.page_chars == [10, 0]


def test_the_result_can_be_stored_as_plain_data(tmp_path):
    document = FakeDocument([FakeItem("text", "hello", page=1)])
    result = ex.extract_document(tmp_path / "p.pdf", intake("text", pages=1), converter_factory(FakeConverter(document), []))

    from dataclasses import asdict
    assert json.loads(json.dumps(asdict(result)))["page_chars"] == [5]


def test_a_table_block_carries_its_caption():
    blocks, _, _ = ex.blocks_from_document(FakeDocument([FakeCaptionedTable("  Table 3: precision of each model ", page=4)]))
    assert blocks[0]["label"] == "table" and blocks[0]["caption"] == "Table 3: precision of each model"


def test_a_table_without_a_readable_caption_still_comes_through():
    blocks, found, failed = ex.blocks_from_document(FakeDocument([
        FakeCaptionedTable(RuntimeError("no caption"), page=2), FakeTable(page=3)]))
    assert [b["caption"] for b in blocks] == ["", ""] and found == 2 and failed == []


def test_only_table_blocks_have_a_caption_key():
    blocks, _, _ = ex.blocks_from_document(FakeDocument([FakeItem("text", "words", page=1)]))
    assert "caption" not in blocks[0]


# ---------- an image file: enlarged, then read whole by Tesseract ----------

def grey_image(width, height=10, color=200):
    from PIL import Image

    return Image.new("RGB", (width, height), (color, color, color))


@pytest.mark.parametrize("width,expected", [(620, 2480), (1000, 2000), (1500, 3000), (2000, 2000), (2600, 2600), (10, 40)])
def test_a_narrow_image_is_enlarged_towards_two_thousand_pixels_but_never_more_than_four_times(width, expected):
    enlarged = ex.enlarged_for_ocr(grey_image(width))

    assert enlarged.width == expected and enlarged.mode == "L"


def test_paragraphs_are_split_on_blank_lines_and_a_word_split_across_lines_is_joined():
    text = "Lung cancer screen-\ning is useful.\nIt takes seconds.\n\n\nBenefits of screening\n\n \nQuit smoking!\n"

    assert ex.paragraphs_of(text) == ["Lung cancer screening is useful. It takes seconds.", "Benefits of screening", "Quit smoking!"]
    assert ex.paragraphs_of("") == [] and ex.paragraphs_of(None) == []


class FakeTesseract:
    """Records the command lines and answers like Tesseract: the text of each page, in order."""

    def __init__(self, texts, returncode=0, stderr=""):
        self.texts, self.returncode, self.stderr, self.calls = list(texts), returncode, stderr, []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        text = self.texts.pop(0) if self.texts else ""
        return SimpleNamespace(returncode=self.returncode, stdout=text, stderr=self.stderr)


def test_every_page_is_read_whole_in_english_and_each_paragraph_is_a_text_block(tmp_path):
    path = tmp_path / "page.png"
    grey_image(600, 800).save(path)
    tesseract = FakeTesseract(["Title line\n\nFirst paragraph here.\nStill the first.\n\nSecond paragraph."])

    blocks, pages = ex.ocr_image_blocks(path, "/usr/bin/tesseract", run=tesseract)

    command = tesseract.calls[0][0]
    assert command[0] == "/usr/bin/tesseract" and command[2:] == ["stdout", "-l", "eng", "--psm", "3"]
    assert pages == 1 and [b["page"] for b in blocks] == [1, 1, 1] and {b["label"] for b in blocks} == {"text"}
    assert [b["text"] for b in blocks] == ["Title line", "First paragraph here. Still the first.", "Second paragraph."]


def test_a_multi_page_image_gives_the_page_number_of_each_block(tmp_path):
    from PIL import Image

    path = tmp_path / "scan.tif"
    first, second = grey_image(500, 700, 255), grey_image(500, 700, 250)
    first.save(path, save_all=True, append_images=[second])
    tesseract = FakeTesseract(["Page one text.", "Page two text."])

    blocks, pages = ex.ocr_image_blocks(path, "tess", run=tesseract)

    assert pages == 2 and [(b["page"], b["text"]) for b in blocks] == [(1, "Page one text."), (2, "Page two text.")]


def test_the_image_given_to_tesseract_is_the_enlarged_greyscale_copy(tmp_path):
    from PIL import Image

    path = tmp_path / "small.jpg"
    grey_image(620, 826).save(path)
    seen = {}

    def inspecting(command, **kwargs):
        with Image.open(command[1]) as sent:
            seen["size"], seen["mode"] = sent.size, sent.mode
        return SimpleNamespace(returncode=0, stdout="text", stderr="")

    ex.ocr_image_blocks(path, "tess", run=inspecting)

    assert seen == {"size": (2480, 3304), "mode": "L"}


def test_a_tesseract_failure_is_an_extraction_error_with_the_message(tmp_path):
    path = tmp_path / "page.png"
    grey_image(600, 800).save(path)

    with pytest.raises(ex.ExtractionError, match="page 1.*bad image"):
        ex.ocr_image_blocks(path, "tess", run=FakeTesseract([""], returncode=1, stderr="bad image"))
