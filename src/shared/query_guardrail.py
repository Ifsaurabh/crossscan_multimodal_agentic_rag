"""Input guardrail.

Also the home of the two shared engines that output_guardrail and
ingestion_guardrails import (the same way ingestion already imported from
query_guardrail): Presidio for PII/credentials and Llama Prompt Guard 2 for
prompt injection. Both load lazily, once per process.

Regex rules that remain: injection COMMANDS in a user's query (here), medical-advice framing
(here) and citation verification (output_guardrail). Prompt Guard is weak on direct requests
to the assistant (see reports/prompt_guard_eval_2026-10-01.md), so the regex is what blocks
queries; the model only flags them, and scores retrieved/ingested text and answers.
"""
import logging
import re
import threading
from typing import Optional

from dotenv import load_dotenv

from shared import langfuse_client

load_dotenv()  # HF_TOKEN for the gated Prompt Guard weights

log = logging.getLogger(__name__)

# ---------- PII / credentials (Presidio) ----------
# Pattern-based entities only. PERSON / LOCATION / DATE_TIME / URL are left out
# on purpose: the corpus is research papers, and redacting author names, place
# names or model names would damage the retrieved text and the answers.
# PHONE_NUMBER is left out too: Presidio can only score a bare phone match at 0.4,
# and lowering the threshold to catch it also redacts "1234567890 samples" and
# "vol 12 (3) 1234-5678" - the same false positive that got the old phone regex
# removed (see test_redact_pii_does_not_false_match_citation_numbers).
SPACY_MODEL = "en_core_web_sm"
PII_SCORE_THRESHOLD = 0.6
PII_PLACEHOLDERS = {
    "EMAIL_ADDRESS": "[REDACTED_EMAIL]",
    "CREDIT_CARD": "[REDACTED_CARD]",
    "US_SSN": "[REDACTED_SSN]",
    "IP_ADDRESS": "[REDACTED_IP]",
    "IBAN_CODE": "[REDACTED_IBAN]",
    "CREDENTIAL": "[REDACTED_CREDENTIAL]",
}

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

_engine_lock = threading.Lock()
_pii_engines = None


def _get_pii_engines():
    global _pii_engines
    with _engine_lock:
        if _pii_engines is None:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine

            nlp_engine = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
            }).create_engine()
            registry = RecognizerRegistry(supported_languages=["en"])
            registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])
            registry.add_recognizer(PatternRecognizer(
                supported_entity="CREDENTIAL",
                patterns=[Pattern(f"credential_{i}", p, 0.9) for i, p in enumerate(CREDENTIAL_PATTERNS)],
            ))
            analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])
            _pii_engines = (analyzer, AnonymizerEngine())
        return _pii_engines


def redact(text: str):
    """Returns (redacted_text, {entity_type: count}). Deterministic, no LLM call."""
    if not text:
        return text, {}
    from presidio_anonymizer.entities import OperatorConfig

    analyzer, anonymizer = _get_pii_engines()
    found = analyzer.analyze(
        text=text, language="en", entities=list(PII_PLACEHOLDERS),
        score_threshold=PII_SCORE_THRESHOLD,
    )
    if not found:
        return text, {}
    result = anonymizer.anonymize(
        text=text, analyzer_results=found,
        operators={e: OperatorConfig("replace", {"new_value": p}) for e, p in PII_PLACEHOLDERS.items()},
    )
    counts = {}
    for item in result.items:
        counts[item.entity_type] = counts.get(item.entity_type, 0) + 1
    return result.text, counts


def redact_pii(text: str):
    """Same (text, count) signature as the old regex version; the count covers
    every entity type, credentials included."""
    cleaned, counts = redact(text)
    return cleaned, sum(counts.values())


# ---------- Prompt injection (Llama Prompt Guard 2) ----------
# 86M over 22M: on 58 varied attacks it blocked 24 vs 6 (and catches other languages), at the price of
# 2 false blocks on 40 tricky legitimate questions (22M: 0) and ~3x the latency.
PROMPT_GUARD_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
# Two thresholds on the model's "malicious" probability: flagging is cheap and
# only annotates; blocking is reserved for near-certain attacks, so a false
# positive on a legitimate question (the corpus includes an AI-security paper)
# is shown as a note instead of refusing the user.
INJECTION_FLAG_THRESHOLD = 0.5
INJECTION_BLOCK_THRESHOLD = 0.95
# Whether a high model score may BLOCK a user's query. Off: the 86M model scored two legitimate
# questions ("Explain how jailbreak attacks bypass safety filters", "Please disregard papers
# before 2020...") as high as real attacks, so no threshold separates them. The score only flags.
MODEL_BLOCKS_QUERIES = False
PROMPT_GUARD_MAX_TOKENS = 512  # the model's context window; longer text is scored in windows

_prompt_guard = None
_prompt_guard_failed = False


def _get_prompt_guard():
    """(tokenizer, model) or None when the weights can't be loaded (no HF_TOKEN /
    licence not accepted / offline). The failure is remembered, not retried per call."""
    global _prompt_guard, _prompt_guard_failed
    with _engine_lock:
        if _prompt_guard is None and not _prompt_guard_failed:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(PROMPT_GUARD_MODEL)
                model = AutoModelForSequenceClassification.from_pretrained(PROMPT_GUARD_MODEL).eval()
                _prompt_guard = (tokenizer, model)
            except Exception as exc:
                _prompt_guard_failed = True
                log.warning("Prompt Guard unavailable (%s: %s) - injection checks are skipped",
                            type(exc).__name__, str(exc)[:200])
        return _prompt_guard


def injection_score(text: str) -> Optional[float]:
    """Probability (0-1) that the text is an injection/jailbreak: the highest
    score over 512-token windows. None when the model isn't available."""
    if not text or not text.strip():
        return 0.0
    guard = _get_prompt_guard()
    if guard is None:
        return None
    import torch

    tokenizer, model = guard
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    window = PROMPT_GUARD_MAX_TOKENS - 2  # room for the special tokens
    pieces = [tokenizer.decode(ids[i:i + window]) for i in range(0, max(len(ids), 1), window)]
    batch = tokenizer(pieces, return_tensors="pt", padding=True, truncation=True,
                      max_length=PROMPT_GUARD_MAX_TOKENS)
    with torch.no_grad():
        probs = torch.softmax(model(**batch).logits, dim=-1)
    return float(probs[:, 1].max())  # label 1 = malicious


PROMPT_GUARD_BATCH = 16  # windows per forward pass when many texts are scored together


def injection_scores(texts: list, batch_size: int = PROMPT_GUARD_BATCH) -> list:
    """The injection score of every text (the same number injection_score gives, up to padding noise), with the 512-token
    windows of ALL the texts scored together in batches of `batch_size`, instead of one text at a time. Used by the ingestion
    of a document, which scores every section and table. An empty text scores 0.0; every score is None when the model
    is not available."""
    guard = _get_prompt_guard()
    if guard is None:
        return [None] * len(texts)
    import torch

    tokenizer, model = guard
    window = PROMPT_GUARD_MAX_TOKENS - 2  # room for the special tokens
    scores, pieces, owners = [0.0] * len(texts), [], []
    for owner, text in enumerate(texts):
        if not text or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        for i in range(0, max(len(ids), 1), window):
            pieces.append(tokenizer.decode(ids[i:i + window]))
            owners.append(owner)
    for start in range(0, len(pieces), batch_size):
        batch = tokenizer(pieces[start:start + batch_size], return_tensors="pt", padding=True, truncation=True,
                          max_length=PROMPT_GUARD_MAX_TOKENS)
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits, dim=-1)[:, 1].tolist()  # label 1 = malicious
        for owner, probability in zip(owners[start:start + batch_size], probs):
            scores[owner] = max(scores[owner], float(probability))
    return scores


def count_guard_tokens(text: str) -> Optional[int]:
    """How many tokens Prompt Guard's tokenizer makes of the text (what injection_score reads), or None when the
    model is not available. Used to report the tokens a document costs the guardrails."""
    if not text or not text.strip():
        return 0
    guard = _get_prompt_guard()
    if guard is None:
        return None
    return len(guard[0](text, add_special_tokens=False)["input_ids"])


def classify_injection(text: str) -> dict:
    return classify_score(injection_score(text))


def classify_score(score: Optional[float]) -> dict:
    """What an injection score means: flagged, blocked, or (score None) the model was not available."""
    return {
        "score": score,
        "available": score is not None,
        "flagged": score is not None and score >= INJECTION_FLAG_THRESHOLD,
        "blocked": score is not None and score >= INJECTION_BLOCK_THRESHOLD,
    }


# ---------- Injection commands (regex) ----------
# Only COMMANDS aimed at the assistant are matched, never mere mentions: the corpus includes an
# AI-security paper, so "what is a system prompt leak?" or "how does jailbreaking work?" are
# legitimate questions and must pass, as must "Act as a summariser: ..." or "You are now reading
# the results section". Role-play phrases therefore only count at the start of a sentence/clause,
# and only when followed by an unrestricted/different persona.
_CLAUSE_START = r"(?:^|[.!?;:\n]\s*|\band\s+|\bthen\s+)(?:please\s+)?"
_PERSONA = r"(?:unrestricted|unfiltered|uncensored|jailbroken|evil|malicious|different|new|dan\b)"
INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above)",
    _CLAUSE_START + r"you are now (?:(?:a|an)\s+)?(?:" + _PERSONA + r"|free\b|no longer\b)",
    r"new instructions?:",
    r"\b(?:reveal|show|print|repeat|display|output|leak|tell me)\s+(?:me\s+)?(?:your\s+(?:system prompt|instructions|prompt)|the system prompt)\b",
    _CLAUSE_START + r"act as (?:if you (?:are|were)\b|(?:an?\s+)?" + _PERSONA + r")",
    _CLAUSE_START + r"pretend (?:you are|to be)\b",
    r"\b(?:enable|enter|activate|switch to)\s+(?:dan|developer|jailbreak|god)\s+mode\b",
    r"\byou are (?:now )?(?:dan|jailbroken)\b",
]
INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(text: str) -> bool:
    return bool(INJECTION_REGEX.search(text))


# ---------- Medical-advice framing (regex) ----------
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


def detect_medical_advice_framing(text: str) -> bool:
    return bool(MEDICAL_ADVICE_REGEX.search(text))


def check_input(query: str) -> dict:
    """Input Guardrail. No LLM call - runs before anything else touches the
    query (including the cache check), so PII never gets used as a cache key or
    sent anywhere. Injection is scored on the ORIGINAL text, so a redaction
    placeholder can't hide an attack."""
    # Spans carry counts, flags and scores only: the query holds personal data until it is redacted.
    with langfuse_client.span("guardrail_presidio") as span:
        cleaned_query, pii_redactions = redact_pii(query)
        span.update(metadata={"redactions": pii_redactions})

    with langfuse_client.span("guardrail_injection_regex") as span:
        regex_hit = detect_prompt_injection(query)
        span.update(metadata={"hit": bool(regex_hit)})
    # A regex hit already blocks, so the model (~0.3 s) is only consulted for the rest.
    if regex_hit:
        injection = {"score": None, "available": True, "flagged": False, "blocked": False}
    else:
        with langfuse_client.span("guardrail_prompt_guard") as span:
            injection = classify_injection(query)
            span.update(metadata={"score": injection.get("score"), "available": injection.get("available"),
                                  "flagged": injection.get("flagged")})
    blocked = regex_hit or (MODEL_BLOCKS_QUERIES and injection["blocked"])
    medical_advice_detected = detect_medical_advice_framing(query)

    reasons = []
    if blocked:
        reasons.append("prompt_injection_detected")
    elif injection["flagged"]:
        reasons.append("prompt_injection_suspected")  # flagged only: the query still goes through
    if medical_advice_detected:
        reasons.append("medical_advice_framing_detected")
    if pii_redactions:
        reasons.append("pii_redacted")

    return {
        "cleaned_query": cleaned_query,
        "blocked": blocked,
        "pii_redactions": pii_redactions,
        "injection_detected": blocked,
        "injection_suspected": injection["flagged"],
        "injection_score": injection["score"],
        "injection_check_available": injection["available"],
        "medical_advice_detected": medical_advice_detected,
        "reasons": reasons,
    }
