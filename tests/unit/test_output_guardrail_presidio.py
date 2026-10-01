"""output_guardrail: Presidio redaction + regex leak/citation checks. Prompt Guard is NOT used on answers
(it missed real leaks and flagged a correct answer - reports/prompt_guard_eval_2026-10-01.md). The old
test_output_guardrail*.py cover redaction (emails, credentials) and citations; this file covers what Presidio added."""
import output_guardrail as og
import query_guardrail as qg


def test_the_output_check_never_calls_the_model(monkeypatch):
    def boom(text):
        raise AssertionError("Prompt Guard must not run on answers")
    monkeypatch.setattr(qg, "injection_score", boom)
    monkeypatch.setattr(qg, "classify_injection", boom)
    result = og.check_output("Sure! My system prompt is: you are a research assistant...", [])
    assert result["injection_leak_detected"] is True and "possible_injection_leak" in result["flags"]


def test_a_correct_answer_quoting_an_attack_payload_is_not_flagged():
    answer = ("The paper describes attacks such as instruction override, where a payload like "
              "'ignore previous instructions' is embedded in a retrieved document [2602.10481v1.pdf, p. 4].")
    chunks = [{"source_pdf": "2602.10481v1.pdf", "page_start": 4, "page_end": 4}]
    result = og.check_output(answer, chunks)
    assert result["flags"] == [] and result["is_grounded"] is True


def test_more_than_emails_is_redacted_in_answers():
    result = og.check_output("Card 4111 1111 1111 1111 and ip 10.0.0.12, mail a@b.org.", [])
    assert result["pii_redactions"] == 3 and result["credential_redactions"] == 0
    assert result["cleaned_answer"] == "Card [REDACTED_CARD] and ip [REDACTED_IP], mail [REDACTED_EMAIL]."


def test_the_result_keys_are_the_same_as_the_old_guardrail():
    assert set(og.check_output("plain answer", [])) == {
        "cleaned_answer", "pii_redactions", "credential_redactions", "verified_citations",
        "unverified_citations", "injection_leak_detected", "clinical_overstatement_detected",
        "flags", "is_grounded",
    }
