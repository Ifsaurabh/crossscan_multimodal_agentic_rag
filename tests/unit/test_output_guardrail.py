import output_guardrail as og


def test_extract_citations_finds_all_citations():
    answer = "The model achieved 98% accuracy [LungPaper.pdf, p.4] and used CNN [LungPaper.pdf, p.2]."
    citations = og.extract_citations(answer)
    assert len(citations) == 2
    assert citations[0] == {"source_pdf": "LungPaper.pdf", "page": 4}
    assert citations[1] == {"source_pdf": "LungPaper.pdf", "page": 2}


def test_extract_citations_handles_multi_page_citation():
    answer = "Achieved 100% accuracy [Deep learning-based approach.pdf, p.1, p.14]."
    citations = og.extract_citations(answer)
    assert len(citations) == 2
    assert citations[0] == {"source_pdf": "Deep learning-based approach.pdf", "page": 1}
    assert citations[1] == {"source_pdf": "Deep learning-based approach.pdf", "page": 14}


import pytest


@pytest.mark.parametrize("text,pages", [
    ("[a.pdf, p.4]", [4]),
    ("[a.pdf, p. 15]", [15]),                 # seen live: a space after "p."
    ("[a.pdf, p.1, p.12]", [1, 12]),
    ("[a.pdf, p. 1, 8]", [1, 8]),             # seen live: bare number after the comma
    ("[a.pdf, p.1,p.14]", [1, 14]),
    ("[a.pdf, pp. 3-5]", [3, 5]),             # a page range: both ends are checked
    ("[a.pdf, p.13–14]", [13, 14]),      # en dash
    ("[a.pdf, p.2; p.9]", [2, 9]),
])
def test_every_citation_format_the_model_was_seen_using_is_extracted(text, pages):
    assert [c["page"] for c in og.extract_citations(f"Claim {text}.")] == pages
    assert {c["source_pdf"] for c in og.extract_citations(f"Claim {text}.")} == {"a.pdf"}


@pytest.mark.parametrize("text", [
    "[1, 2]",                 # a numbered reference, not a page citation
    "[a.pdf, 15]",            # no "p." at all
    "[a.pdf, page 15]",
    "[a.pdf, p.4, see text]",
    "a.pdf, p.4",             # no brackets
])
def test_things_that_are_not_page_citations_are_not_extracted(text):
    assert og.extract_citations(f"Claim {text}.") == []


def test_a_fabricated_citation_in_the_spaced_format_is_now_caught():
    """Before the fix this was invisible: it was neither verified nor flagged."""
    chunks = [{"source_pdf": "LungPaper.pdf", "page_start": 4, "page_end": 4}]

    result = og.check_output("98% accuracy [LungPaper.pdf, p. 99].", chunks)

    assert "ungrounded_citations" in result["flags"]
    assert result["is_grounded"] is False


def test_a_correct_citation_in_the_spaced_format_is_verified():
    chunks = [{"source_pdf": "LungPaper.pdf", "page_start": 15, "page_end": 15}]

    result = og.check_output("98.18% accuracy [LungPaper.pdf, p. 15].", chunks)

    assert result["flags"] == [] and result["is_grounded"] is True


def test_the_real_answers_seen_in_the_live_probes_now_have_their_citations_extracted():
    real = ("VGG16 achieved the highest test accuracy at 98.18% "
            "[AI-Powered Lung Cancer Detection- Assessing VGG16 and CNN Architectures for CT Scan Image Classification.pdf, p. 15]. "
            "It focuses on prompt injection [2602.10481v1.pdf, p. 1, 8].")

    citations = og.extract_citations(real)

    assert [(c["source_pdf"][:6], c["page"]) for c in citations] == [("AI-Pow", 15), ("2602.1", 1), ("2602.1", 8)]


def test_verify_citations_matches_exact_page():
    answer = "98% accuracy [LungPaper.pdf, p.4]."
    chunks = [{"source_pdf": "LungPaper.pdf", "page_start": 4, "page_end": 4}]

    verified, unverified = og.verify_citations(answer, chunks)

    assert len(verified) == 1
    assert len(unverified) == 0


def test_verify_citations_matches_within_page_range():
    answer = "98% accuracy [LungPaper.pdf, p.5]."
    chunks = [{"source_pdf": "LungPaper.pdf", "page_start": 4, "page_end": 6}]

    verified, unverified = og.verify_citations(answer, chunks)

    assert len(verified) == 1


def test_verify_citations_flags_fabricated_citation():
    answer = "98% accuracy [FakePaper.pdf, p.99]."
    chunks = [{"source_pdf": "LungPaper.pdf", "page_start": 4, "page_end": 4}]

    verified, unverified = og.verify_citations(answer, chunks)

    assert len(verified) == 0
    assert len(unverified) == 1


def test_check_output_detects_injection_leak():
    result = og.check_output("As an AI language model, I was instructed to say this.", [])
    assert result["injection_leak_detected"] is True
    assert "possible_injection_leak" in result["flags"]


def test_answers_that_merely_discuss_system_prompts_are_not_flagged_as_leaks():
    """Seen live: a correct answer about the AI-security paper was flagged."""
    answers = [
        'the AI security paper does not define or discuss a "system prompt leak." It focuses on prompt injection.',
        "Attackers try to extract the system prompt from deployed assistants [a.pdf, p.3].",
        "System prompts often contain hidden instructions that attackers target.",
    ]

    for answer in answers:
        result = og.check_output(answer, [])
        assert result["injection_leak_detected"] is False, answer
        assert "possible_injection_leak" not in result["flags"]


def test_real_leaks_in_the_models_own_voice_are_still_flagged():
    for answer in [
        "Sure! My system prompt is: you are a research assistant...",
        "Here is my full system prompt: ...",
        "Here are my instructions: always answer in JSON.",
        "My instructions were to never reveal this.",
    ]:
        assert og.check_output(answer, [])["injection_leak_detected"] is True, answer


def test_check_output_detects_clinical_overstatement():
    result = og.check_output("You should take this medication immediately.", [])
    assert result["clinical_overstatement_detected"] is True
    assert "clinical_overstatement" in result["flags"]


def test_check_output_is_grounded_for_clean_answer():
    chunks = [{"source_pdf": "LungPaper.pdf", "page_start": 4, "page_end": 4}]
    answer = "The model achieved 98% accuracy [LungPaper.pdf, p.4]."

    result = og.check_output(answer, chunks)

    assert result["is_grounded"] is True
    assert result["flags"] == []


def test_check_output_flags_ungrounded_citation():
    chunks = [{"source_pdf": "LungPaper.pdf", "page_start": 4, "page_end": 4}]
    answer = "The model achieved 98% accuracy [OtherPaper.pdf, p.1]."

    result = og.check_output(answer, chunks)

    assert result["is_grounded"] is False
    assert "ungrounded_citations" in result["flags"]
