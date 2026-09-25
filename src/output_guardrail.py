import re

# Same pattern as query_guardrail.py/ingestion_guardrails.py - last checkpoint
# before the answer reaches the user, so PII gets REDACTED here (not just
# flagged): closes the general-knowledge-answer path, which never goes
# through document/query redaction since it involves no retrieved content.
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Credential/secret shapes - matches actual key formats, not bare words like
# "password" (a security paper discussing password policy in prose is not a
# leak; the AI-security paper in this corpus already contains realistic
# example payloads, so an actual key-shaped string appearing is plausible).
CREDENTIAL_PATTERNS = [
    r"\bsk-[A-Za-z0-9]{20,}\b",                                    # OpenAI-style key
    r"\bAKIA[0-9A-Z]{16}\b",                                       # AWS access key ID
    r"(?:api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9\-_]{16,}['\"]?",
    r"password\s*[:=]\s*['\"]?\S{6,}['\"]?",                       # password=... assignment, not bare mentions
    r"Bearer\s+[A-Za-z0-9\-_.]{20,}",                              # Bearer token
]
CREDENTIAL_REGEX = re.compile("|".join(CREDENTIAL_PATTERNS), re.IGNORECASE)

# Citation format requested by agent_quality_generate.py's prompt:
# "[source_pdf, p.4]". Models drift from it, and every format that is NOT
# recognised here silently escapes verification, so the pattern accepts what
# was actually seen in live answers: "p.4", "p. 4", "p.1, p.14", "p. 1, 8",
# and page ranges such as "pp. 3-5" (both ends are checked).
_PAGE = r"\d+(?:\s*[-–]\s*\d+)?"
CITATION_PATTERN = re.compile(
    rf"\[([^,\]]+),\s*(pp?\.\s*{_PAGE}(?:\s*[,;]\s*(?:pp?\.\s*)?{_PAGE})*)\]"
)
PAGE_NUMBER_PATTERN = re.compile(r"(\d+)")

# Same injection patterns as Stage 7 (src/query_guardrail.py) - defense in
# depth: check the OUTPUT doesn't show signs the model was successfully
# manipulated into ignoring its instructions (e.g. leaking a system prompt).
#
# The model talking about ITS OWN instructions is the signal. The bare words
# "system prompt" are not: the corpus includes an AI-security paper, so a
# correct answer about system-prompt leaks must not raise a false alarm.
INJECTION_LEAK_PATTERNS = [
    r"\bmy system prompt\b",
    r"here (is|are) my (full |complete |exact )?(system prompt|instructions)",
    r"my instructions (are|were)",
    r"as an ai (language )?model,? i (was|am) instructed",
    r"ignoring (previous|prior) instructions",
]
INJECTION_LEAK_REGEX = re.compile("|".join(INJECTION_LEAK_PATTERNS), re.IGNORECASE)

CLINICAL_OVERSTATEMENT_PATTERNS = [
    r"you (should|must) take",
    r"this (proves|confirms) you (have|are)",
    r"i (recommend|advise) (you )?(to )?",
]
CLINICAL_OVERSTATEMENT_REGEX = re.compile("|".join(CLINICAL_OVERSTATEMENT_PATTERNS), re.IGNORECASE)


def extract_citations(answer: str):
    citations = []
    for m in CITATION_PATTERN.finditer(answer):
        source_pdf = m.group(1).strip()
        for page_match in PAGE_NUMBER_PATTERN.finditer(m.group(2)):
            citations.append({"source_pdf": source_pdf, "page": int(page_match.group(1))})
    return citations


def verify_citations(answer: str, retrieved_chunks: list):
    """Reuses the same 'verify against source' pattern proven in Stage 6b's
    entity extraction: check each cited (source_pdf, page) actually
    corresponds to something that was really retrieved, not invented."""
    citations = extract_citations(answer)
    retrieved_sources = {(c.get("source_pdf"), c.get("page_start")) for c in retrieved_chunks}
    # Also allow matching within a chunk's page range, not just exact page_start.
    retrieved_ranges = [
        (c.get("source_pdf"), c.get("page_start"), c.get("page_end")) for c in retrieved_chunks
    ]

    verified = []
    unverified = []
    for citation in citations:
        matched = (citation["source_pdf"], citation["page"]) in retrieved_sources
        if not matched:
            matched = any(
                source == citation["source_pdf"] and start is not None and end is not None
                and start <= citation["page"] <= end
                for source, start, end in retrieved_ranges
            )
        (verified if matched else unverified).append(citation)

    return verified, unverified


def check_output(answer: str, retrieved_chunks: list) -> dict:
    """Stage 10 Output Guardrail. Deterministic, no LLM call."""
    cleaned_answer, pii_redactions = EMAIL_PATTERN.subn("[REDACTED_EMAIL]", answer)
    cleaned_answer, credential_redactions = CREDENTIAL_REGEX.subn("[REDACTED_CREDENTIAL]", cleaned_answer)

    verified, unverified = verify_citations(cleaned_answer, retrieved_chunks)
    injection_leak_detected = bool(INJECTION_LEAK_REGEX.search(cleaned_answer))
    clinical_overstatement_detected = bool(CLINICAL_OVERSTATEMENT_REGEX.search(cleaned_answer))

    flags = []
    if unverified:
        flags.append("ungrounded_citations")
    if injection_leak_detected:
        flags.append("possible_injection_leak")
    if clinical_overstatement_detected:
        flags.append("clinical_overstatement")
    if pii_redactions:
        flags.append("pii_redacted")
    if credential_redactions:
        flags.append("credential_redacted")

    return {
        "cleaned_answer": cleaned_answer,
        "pii_redactions": pii_redactions,
        "credential_redactions": credential_redactions,
        "verified_citations": verified,
        "unverified_citations": unverified,
        "injection_leak_detected": injection_leak_detected,
        "clinical_overstatement_detected": clinical_overstatement_detected,
        "flags": flags,
        "is_grounded": len(unverified) == 0 and len(verified) > 0,
    }
