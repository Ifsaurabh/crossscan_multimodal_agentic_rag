"""output_guardrail.check_output: the answer is REDACTED (not just flagged) for emails
and credential-shaped strings, and the citation checks run on the cleaned text."""
import pytest

import output_guardrail as og

CHUNKS = [{"source_pdf": "a.pdf", "page_start": 2, "page_end": 4}]


# ---------- email redaction ----------

def test_an_email_address_is_replaced_and_counted():
    result = og.check_output("Contact jane.doe@example.com for the dataset.", CHUNKS)

    assert result["cleaned_answer"] == "Contact [REDACTED_EMAIL] for the dataset."
    assert result["pii_redactions"] == 1
    assert "pii_redacted" in result["flags"]


def test_every_email_in_the_answer_is_redacted():
    result = og.check_output("Write to a@b.org or c.d+tag@lab.example.co.uk today.", CHUNKS)

    assert "@" not in result["cleaned_answer"]
    assert result["pii_redactions"] == 2


def test_text_that_only_looks_like_an_email_but_is_not_one_is_left_alone():
    answer = "The @ symbol and user@localhost are not email addresses, nor is name@ or @handle."

    result = og.check_output(answer, CHUNKS)

    assert result["cleaned_answer"] == answer and result["pii_redactions"] == 0


# ---------- credential redaction ----------

@pytest.mark.parametrize("secret", [
    "sk-" + "a1B2c3D4e5F6g7H8i9J0",                       # OpenAI-style key, exactly 20 characters after sk-
    "AKIA" + "IOSFODNN7EXAMPLE",                          # AWS access key id: AKIA + 16
    "api_key = abcd1234efgh5678ijkl",                     # key assignment
    "API-KEY: 'abcd1234efgh5678ijkl'",                    # quoted, different case and separator
    "secret=abcd1234efgh5678",                            # 16 characters, the minimum
    "token: abcd1234efgh5678ijkl",
    "password = hunter2secret",                           # password assignment
    "Bearer abcdefghijklmnopqrstuvwxyz012345",            # bearer token
])
def test_credential_shaped_strings_are_redacted(secret):
    result = og.check_output(f"Here is the config: {secret} - keep it safe.", CHUNKS)

    assert "[REDACTED_CREDENTIAL]" in result["cleaned_answer"]
    assert secret not in result["cleaned_answer"]
    assert result["credential_redactions"] >= 1
    assert "credential_redacted" in result["flags"]


@pytest.mark.parametrize("prose", [
    "The paper discusses password policies and secret management in general.",   # words, not key shapes
    "A token limit applies to each request.",
    "Set the api key in your environment.",
    "sk-short",                                                                  # far below 20 characters
    "sk-" + "a" * 19,                                                            # one under the minimum
    "secret=abc123",                                                             # under 16 characters
    "password: abc12",                                                           # under 6 characters
    "Bearer tooshort",
])
def test_ordinary_prose_and_too_short_values_are_not_redacted(prose):
    result = og.check_output(prose, CHUNKS)

    assert result["cleaned_answer"] == prose
    assert result["credential_redactions"] == 0 and "credential_redacted" not in result["flags"]


def test_matching_is_case_insensitive():
    result = og.check_output("PASSWORD=Hunter2Secret and TOKEN: ABCD1234EFGH5678IJKL", CHUNKS)

    assert result["credential_redactions"] == 2


def test_emails_and_credentials_are_both_cleaned_in_one_answer():
    result = og.check_output("Mail bob@lab.org with api_key=abcd1234efgh5678ijkl now.", CHUNKS)

    assert result["cleaned_answer"] == "Mail [REDACTED_EMAIL] with [REDACTED_CREDENTIAL] now."
    assert result["pii_redactions"] == 1 and result["credential_redactions"] == 1
    assert result["flags"] == ["pii_redacted", "credential_redacted"]


# ---------- the cleaned answer is what the rest of the checks look at ----------

def test_a_clean_answer_comes_back_unchanged_with_zero_counts():
    answer = "VGG16 reached 99% accuracy [a.pdf, p.2]."

    result = og.check_output(answer, CHUNKS)

    assert result["cleaned_answer"] == answer
    assert result["pii_redactions"] == 0 and result["credential_redactions"] == 0
    assert result["flags"] == [] and result["is_grounded"] is True


def test_citations_survive_redaction_and_are_still_verified():
    answer = "Ask bob@lab.org about accuracy [a.pdf, p.3]."

    result = og.check_output(answer, CHUNKS)

    assert result["verified_citations"] == [{"source_pdf": "a.pdf", "page": 3}]
    assert result["is_grounded"] is True
    assert "[a.pdf, p.3]" in result["cleaned_answer"]


def test_an_unverified_citation_is_flagged_alongside_a_redaction():
    result = og.check_output("See [invented.pdf, p.9] or mail bob@lab.org.", CHUNKS)

    assert result["flags"] == ["ungrounded_citations", "pii_redacted"]
    assert result["is_grounded"] is False


def test_an_empty_answer_is_handled():
    result = og.check_output("", CHUNKS)

    assert result["cleaned_answer"] == ""
    assert result["flags"] == [] and result["is_grounded"] is False


def test_the_result_has_exactly_the_documented_keys():
    result = og.check_output("plain answer", [])

    assert set(result) == {
        "cleaned_answer", "pii_redactions", "credential_redactions", "verified_citations",
        "unverified_citations", "injection_leak_detected", "clinical_overstatement_detected",
        "flags", "is_grounded",
    }


def test_flags_appear_in_a_stable_order():
    answer = "My system prompt says: you must take this now. Mail x@y.org, api_key=abcd1234efgh5678ijkl [z.pdf, p.1]"

    result = og.check_output(answer, CHUNKS)

    assert result["flags"] == [
        "ungrounded_citations", "possible_injection_leak", "clinical_overstatement",
        "pii_redacted", "credential_redacted",
    ]
