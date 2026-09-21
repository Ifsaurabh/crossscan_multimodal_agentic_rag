import re

# Same pattern as src/ingestion_guardrails.py (Stage 3b) - deterministic,
# no LLM call, so PII never reaches any LLM in the first place.
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Common prompt-injection phrasings. Not exhaustive, but catches the
# well-known patterns without needing an LLM call to check for them.
#
# Only COMMANDS aimed at the assistant are matched, never mere mentions: the
# corpus includes an AI-security paper, so "what is a system prompt leak?" or
# "how does jailbreaking work?" are legitimate questions and must pass, as
# must "the layers act as a feature extractor". Role-play phrases therefore
# only count at the start of a sentence/clause.
_CLAUSE_START = r"(?:^|[.!?;:\n]\s*|\band\s+|\bthen\s+)(?:please\s+)?"
INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above)",
    _CLAUSE_START + r"you are now\b",
    r"new instructions?:",
    r"\b(?:reveal|show|print|repeat|display|output|leak|tell me)\s+(?:me\s+)?(?:your\s+(?:system prompt|instructions|prompt)|the system prompt)\b",
    _CLAUSE_START + r"act as (?:if you (?:are|were)|an?)\s",
    _CLAUSE_START + r"pretend (?:you are|to be)\b",
    r"\b(?:enable|enter|activate|switch to)\s+(?:dan|developer|jailbreak|god)\s+mode\b",
    r"\byou are (?:now )?(?:dan|jailbroken)\b",
]
INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

MEDICAL_ADVICE_PATTERNS = [
    r"should i (?:take|start|stop) (?:any |this |these |my |the |a )?(?:medication|medicine|drug|pill|dose|dosage|treatment|chemo|chemotherapy)",
    r"what (medication|drug|dosage|dose) should",
    r"diagnos(e|is) me",
    r"am i (going to die|dying|terminal)",
    r"is my (tumor|cancer|condition)",
    r"treatment for my",
]
MEDICAL_ADVICE_REGEX = re.compile("|".join(MEDICAL_ADVICE_PATTERNS), re.IGNORECASE)

# Raised in the graph state when a user asks for personal medical advice, so the
# answer can carry a safety note (see retrieval_graph.MEDICAL_NOTE).
MEDICAL_ADVICE_FLAG = "medical_advice_framing"


def redact_pii(text: str):
    redactions = 0

    def replace_email(match):
        nonlocal redactions
        redactions += 1
        return "[REDACTED_EMAIL]"

    text = EMAIL_PATTERN.sub(replace_email, text)
    return text, redactions


def detect_prompt_injection(text: str) -> bool:
    return bool(INJECTION_REGEX.search(text))


def detect_medical_advice_framing(text: str) -> bool:
    return bool(MEDICAL_ADVICE_REGEX.search(text))


def check_input(query: str) -> dict:
    """Stage 7 Input Guardrail. Deterministic, no LLM call - runs before
    anything else touches the query (including the cache check), so PII
    never gets used as a cache key or sent anywhere."""
    cleaned_query, pii_redactions = redact_pii(query)

    injection_detected = detect_prompt_injection(query)
    medical_advice_detected = detect_medical_advice_framing(query)

    blocked = injection_detected
    reasons = []
    if injection_detected:
        reasons.append("prompt_injection_detected")
    if medical_advice_detected:
        reasons.append("medical_advice_framing_detected")
    if pii_redactions:
        reasons.append("pii_redacted")

    return {
        "cleaned_query": cleaned_query,
        "blocked": blocked,
        "pii_redactions": pii_redactions,
        "injection_detected": injection_detected,
        "medical_advice_detected": medical_advice_detected,
        "reasons": reasons,
    }
