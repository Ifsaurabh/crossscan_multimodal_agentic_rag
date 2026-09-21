"""The ONE place the application talks to language models.

Every runtime LLM call (agents, conversation summary, DeepEval/RAGAS judges)
goes through `generate()` and names either a TIER or a TASK (which maps to a
tier). This module owns:
  - the tiers: ordered lists of (provider, model) - a primary and at least one
    fallback each - edit TIERS below to change models or providers
  - API keys, read ONLY from environment variables / .env, never stored here
  - automatic fallback WITHIN a tier: a model whose provider has no key/SDK is
    skipped, a model that fails (quota, outage, auth) is put on a short
    cooldown, and the tier's next model answers instead
  - the rule that a tier NEVER borrows another tier's models: if every model
    in a tier fails, AllProvidersFailed is raised

Tiers:
  fast        simple tasks and simple queries - cheap and fast models
  reasoning   complex queries and reasoning - smarter models
  evaluation  DeepEval / RAGAS judges

Callers never import a provider SDK and never know which model answered
unless they look at LLMResult.
"""
import os
import threading
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

import usage_tracker
from retrieval_config import GEMINI_MODEL

load_dotenv()

# ============================== SETTINGS ==============================
# Connection details per provider. API KEYS are never stored here: only the
# NAME of the environment variable that holds each key.
PROVIDERS = {
    "gemini": {"api_key_env": "GEMINI_API_KEY"},
    "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "timeout_seconds": 60, "max_output_tokens": 4096},
    "openai": {"api_key_env": "OPENAI_API_KEY", "timeout_seconds": 60, "max_output_tokens": 4096},
}

# Ordered (provider, model) lists: first = primary, the rest = fallbacks, tried
# in order. Every tier needs at least MIN_MODELS_PER_TIER distinct models.
# Google models only for now; to move a tier to another vendor later, change
# its entries here (e.g. {"provider": "openai", "model": "..."}) and add the
# key to .env. Free-tier quotas are PER MODEL, so a different fallback model
# also has its own daily allowance (a model listed in two tiers shares one).
TIERS = {
    "fast": [
        {"provider": "gemini", "model": GEMINI_MODEL},  # gemini-3.6-flash, defined in retrieval_config.py
        {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
    ],
    "reasoning": [
        {"provider": "gemini", "model": "gemini-3.8-flash"},
        {"provider": "gemini", "model": "gemini-3.7-flash"},
    ],
    "evaluation": [
        {"provider": "gemini", "model": "gemini-3.7-flash"},
        {"provider": "gemini", "model": "gemini-3.5-flash"},
    ],
}

# Which tier each task uses. Re-tiering a task is a one-line change here.
# Answer writing is chosen per question: the query planner labels every
# sub-question "simple" or "complex" and the answer writer uses the matching
# task below.
TASK_TIERS = {
    "query_planner": "fast",
    "quality_check": "fast",
    "general_knowledge": "fast",
    "conversation_summary": "fast",
    "answer_simple": "fast",
    "answer_complex": "reasoning",
    "deepeval_judge": "evaluation",
    "ragas_judge": "evaluation",
    "online_judge": "evaluation",  # scores a sample of live answers in production
}

MIN_MODELS_PER_TIER = 2

# Attempts per model when Gemini answers "busy" (503). With a fallback model
# waiting in the tier one retry is enough (one 2 s wait, then switch); the last
# usable model of a tier keeps the full 4 attempts (waits of 2, 4 and 8 s).
FAILOVER_RETRIES = 2
FULL_RETRIES = 4

# After a model fails, skip it for this long instead of retrying it on every
# call. Quota errors get the longest cooldown (free tiers reset daily), then
# bad credentials, then transient outages.
COOLDOWN_SECONDS = {"quota": 900, "auth": 3600, "unavailable": 60, "other": 0}
# ======================================================================


class ProviderUnavailable(Exception):
    """A provider cannot be used at all right now (SDK not installed, ...)."""


class EmptyResponse(Exception):
    """The provider answered but returned no text (e.g. a safety block)."""


class AllProvidersFailed(Exception):
    """Every model in the requested tier failed or was unavailable."""

    def __init__(self, attempts: list, tier: str = None):
        self.attempts = attempts
        self.tier = tier
        details = "; ".join(
            f"{a['provider']}:{a.get('model', '?')} {a['status']} ({a.get('detail', '')})" for a in attempts
        ) or "no models configured"
        super().__init__(f"No model in tier '{tier}' could answer. {details}")


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    tier: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    attempts: list = field(default_factory=list)  # models tried BEFORE the one that answered

    @property
    def fallback_used(self) -> bool:
        return any(a["status"] == "failed" for a in self.attempts)


def validate_settings() -> None:
    """Fails fast on a broken configuration (called at import)."""
    for tier, models in TIERS.items():
        if len(models) < MIN_MODELS_PER_TIER:
            raise ValueError(f"Tier '{tier}' needs at least {MIN_MODELS_PER_TIER} models (a primary and a fallback).")
        seen = set()
        for spec in models:
            if spec["provider"] not in PROVIDERS:
                raise ValueError(f"Tier '{tier}' uses unknown provider '{spec['provider']}'.")
            key = (spec["provider"], spec["model"])
            if key in seen:
                raise ValueError(f"Tier '{tier}' lists {spec['provider']}:{spec['model']} twice; a fallback must be a different model.")
            seen.add(key)
    for task, tier in TASK_TIERS.items():
        if tier not in TIERS:
            raise ValueError(f"Task '{task}' maps to unknown tier '{tier}'.")


validate_settings()

_lock = threading.Lock()
_clients = {}
_cooldown_until = {}


# ------------------------------ helpers ------------------------------

def classify_error(error: Exception) -> str:
    """'quota' | 'auth' | 'unavailable' | 'other' - decides the cooldown."""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    text = f"{type(error).__name__} {error}".lower()

    if code == 429 or any(w in text for w in ("resource_exhausted", "rate limit", "ratelimit", "quota", "too many requests")):
        return "quota"
    if code in (401, 403) or any(w in text for w in ("authentication", "unauthorized", "permission", "invalid api key", "api key not valid", "invalid x-api-key")):
        return "auth"
    if code in (500, 502, 503, 504, 529) or any(w in text for w in ("timeout", "timed out", "connection", "unavailable", "overloaded")):
        return "unavailable"
    return "other"


def _cooldown_key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _start_cooldown(provider: str, model: str, kind: str, now: float = None) -> None:
    seconds = COOLDOWN_SECONDS.get(kind, 0)
    if seconds:
        with _lock:
            _cooldown_until[_cooldown_key(provider, model)] = (time.monotonic() if now is None else now) + seconds


def _cooldown_remaining(provider: str, model: str, now: float = None) -> int:
    with _lock:
        until = _cooldown_until.get(_cooldown_key(provider, model), 0)
    return max(0, int(until - (time.monotonic() if now is None else now)))


def reset_cooldowns() -> None:
    with _lock:
        _cooldown_until.clear()


def record_failure(provider: str, model: str, error: Exception) -> str:
    """For callers that use a provider's native client directly (the RAGAS
    judge): puts the model on the same cooldown generate() would and returns
    the error kind."""
    kind = classify_error(error)
    _start_cooldown(provider, model, kind)
    return kind


def _get_client(name: str, factory):
    with _lock:
        if name not in _clients:
            _clients[name] = factory()
        return _clients[name]


def _unavailable_reason(provider: str, model: str, client_override) -> str:
    """Why this model should be skipped right now, or None if it can be tried."""
    remaining = _cooldown_remaining(provider, model)
    if remaining:
        return f"cooling down after a recent failure ({remaining}s left)"
    if provider == "gemini" and client_override is not None:
        return None
    env_name = PROVIDERS[provider]["api_key_env"]
    if not os.environ.get(env_name):
        return f"{env_name} is not set"
    return None


def resolve_tier(tier: str = None, task: str = None) -> str:
    if tier is not None:
        if tier not in TIERS:
            raise ValueError(f"Unknown tier '{tier}'. Tiers: {sorted(TIERS)}")
        return tier
    if task is not None:
        if task not in TASK_TIERS:
            raise ValueError(f"Unknown task '{task}'. Add it to TASK_TIERS. Tasks: {sorted(TASK_TIERS)}")
        return TASK_TIERS[task]
    raise ValueError("generate() needs a tier= or a task=.")


def ready_models(tier: str) -> list:
    """The tier's models that can be tried right now, in order, as
    {provider, model, config}. For callers that need a provider's native
    client (the RAGAS judge)."""
    ready = []
    for spec in TIERS[resolve_tier(tier)]:
        if _unavailable_reason(spec["provider"], spec["model"], None) is None:
            ready.append({**spec, "config": PROVIDERS[spec["provider"]]})
    return ready


def status() -> dict:
    """Per-tier, per-model readiness for health/admin views. No network calls."""
    report = {}
    for tier, models in TIERS.items():
        entries = []
        for spec in models:
            reason = _unavailable_reason(spec["provider"], spec["model"], None)
            entries.append({"provider": spec["provider"], "model": spec["model"], "ready": reason is None, "reason": reason})
        report[tier] = entries
    return report


def config_snapshot() -> dict:
    """Flat settings dict for logging with evaluation runs."""
    return {
        f"llm_tier_{tier}": ",".join(f"{s['provider']}:{s['model']}" for s in models)
        for tier, models in TIERS.items()
    }


# ------------------------------ adapters ------------------------------
# Each adapter: (config, model, system_instruction, user_content, client) -> LLMResult.
# Looked up through _ADAPTERS at call time so tests can replace them.

def _call_gemini(config, model, system_instruction, user_content, client=None) -> LLMResult:
    import gemini_retry  # retry-with-backoff on 503 + opportunistic prompt caching

    if client is None:
        from google import genai

        client = _get_client("gemini", lambda: genai.Client(api_key=os.environ[config["api_key_env"]]))

    retries = config.get("max_retries", gemini_retry.DEFAULT_MAX_RETRIES)
    if system_instruction:
        response = gemini_retry.call_with_cache(client, model, system_instruction, user_content, max_retries=retries)
    else:
        response = gemini_retry.call_with_retry(client, model, user_content, max_retries=retries)

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise EmptyResponse(f"gemini model {model} returned no text")

    meta = getattr(response, "usage_metadata", None)
    # gemini_retry already recorded this call's tokens in usage_tracker.
    return LLMResult(
        text=text, provider="gemini", model=model,
        prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
    )


def _call_anthropic(config, model, system_instruction, user_content, client=None) -> LLMResult:
    try:
        import anthropic
    except ImportError:
        raise ProviderUnavailable("the 'anthropic' package is not installed (pip install anthropic)")

    client = _get_client(
        "anthropic",
        lambda: anthropic.Anthropic(api_key=os.environ[config["api_key_env"]], timeout=config["timeout_seconds"]),
    )
    request = {
        "model": model,
        "max_tokens": config["max_output_tokens"],
        "messages": [{"role": "user", "content": user_content}],
    }
    if system_instruction:
        request["system"] = system_instruction

    response = client.messages.create(**request)
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()
    if not text:
        raise EmptyResponse(f"anthropic model {model} returned no text")

    usage = response.usage
    prompt = getattr(usage, "input_tokens", 0) or 0
    output = getattr(usage, "output_tokens", 0) or 0
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    usage_tracker.record_tokens(prompt, output, cached)
    return LLMResult(text=text, provider="anthropic", model=model,
                     prompt_tokens=prompt, output_tokens=output, cached_tokens=cached)


def _call_openai(config, model, system_instruction, user_content, client=None) -> LLMResult:
    try:
        import openai
    except ImportError:
        raise ProviderUnavailable("the 'openai' package is not installed (pip install openai)")

    client = _get_client(
        "openai",
        lambda: openai.OpenAI(api_key=os.environ[config["api_key_env"]], timeout=config["timeout_seconds"]),
    )
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model, messages=messages, max_completion_tokens=config["max_output_tokens"],
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise EmptyResponse(f"openai model {model} returned no text")

    usage = response.usage
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    output = getattr(usage, "completion_tokens", 0) or 0
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    usage_tracker.record_tokens(prompt, output, cached)
    return LLMResult(text=text, provider="openai", model=model,
                     prompt_tokens=prompt, output_tokens=output, cached_tokens=cached)


_ADAPTERS = {"gemini": _call_gemini, "anthropic": _call_anthropic, "openai": _call_openai}


# ------------------------------ public API ------------------------------

def generate(system_instruction: str, user_content: str, *, tier: str = None, task: str = None, client=None) -> LLMResult:
    """Answer `user_content` under `system_instruction` (may be empty) using
    the first working model of the requested tier.

    Name the tier directly (`tier="fast"`) or the task (`task="query_planner"`,
    looked up in TASK_TIERS). `client` optionally supplies the Gemini client
    (used by tests). Raises AllProvidersFailed if no model in the tier could
    answer - a tier never borrows another tier's models."""
    tier = resolve_tier(tier, task)
    attempts = []
    last_error = None

    for position, spec in enumerate(TIERS[tier]):
        provider, model = spec["provider"], spec["model"]

        reason = _unavailable_reason(provider, model, client)
        if reason:
            attempts.append({"provider": provider, "model": model, "status": "skipped", "detail": reason})
            continue

        # A busy model is retried with backoff, but if a healthy fallback is
        # waiting in this tier there is no point sitting through 2+4+8 s of
        # waits: the fallback is a separate capacity pool. Only the LAST usable
        # model of the tier gets the full patience.
        fallback_ready = any(
            _unavailable_reason(other["provider"], other["model"], client) is None
            for other in TIERS[tier][position + 1:]
        )
        config = {**PROVIDERS[provider], "max_retries": FAILOVER_RETRIES if fallback_ready else FULL_RETRIES}

        try:
            result = _ADAPTERS[provider](
                config, model, system_instruction, user_content,
                client if provider == "gemini" else None,
            )
        except ProviderUnavailable as e:
            attempts.append({"provider": provider, "model": model, "status": "skipped", "detail": str(e)})
            continue
        except Exception as e:
            kind = classify_error(e)
            _start_cooldown(provider, model, kind)
            attempts.append({
                "provider": provider, "model": model, "status": "failed", "kind": kind,
                "detail": f"{type(e).__name__}: {str(e)[:200]}",
            })
            last_error = e
            print(f"   (LLM {provider}:{model} failed [{kind}], trying the next model in tier '{tier}'...)")
            continue

        result.tier = tier
        result.attempts = attempts
        if result.fallback_used:
            print(f"   (tier '{tier}' answered by fallback model {provider}:{model})")
        return result

    raise AllProvidersFailed(attempts, tier) from last_error
