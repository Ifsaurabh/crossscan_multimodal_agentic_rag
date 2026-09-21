import contextvars
import os
import threading

_lock = threading.Lock()
_totals = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

# Per-request accumulator: lets one chat request report exactly what IT
# consumed even while other requests run concurrently (the global totals
# below can't be diffed safely under concurrency).
_request_usage = contextvars.ContextVar("request_usage", default=None)


def _empty() -> dict:
    return {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


def record(response) -> None:
    """Accumulates token usage from a Gemini response (its usage_metadata).
    Tolerates responses without usage data."""
    meta = getattr(response, "usage_metadata", None)
    prompt = getattr(meta, "prompt_token_count", 0) or 0 if meta is not None else 0
    output = getattr(meta, "candidates_token_count", 0) or 0 if meta is not None else 0
    cached = getattr(meta, "cached_content_token_count", 0) or 0 if meta is not None else 0
    record_tokens(prompt, output, cached)


def record_tokens(prompt: int, output: int, cached: int = 0) -> None:
    """Records one model call's token counts (provider-neutral; record() feeds
    Gemini responses through here, the Anthropic/OpenAI adapters call it directly)."""
    with _lock:
        _totals["calls"] += 1
        _totals["prompt_tokens"] += prompt
        _totals["output_tokens"] += output
        _totals["cached_tokens"] += cached

    request = _request_usage.get()
    if request is not None:
        request["calls"] += 1
        request["prompt_tokens"] += prompt
        request["output_tokens"] += output
        request["cached_tokens"] += cached


def begin_request() -> None:
    """Start attributing model usage to the current request (context)."""
    _request_usage.set(_empty())


def end_request() -> dict:
    """Stop attributing and return what the request consumed."""
    usage = _request_usage.get()
    _request_usage.set(None)
    return usage or _empty()


def snapshot() -> dict:
    """Current totals plus an estimated cost. Prices come from
    GEMINI_PRICE_PER_M_INPUT / GEMINI_PRICE_PER_M_OUTPUT (USD per million
    tokens) and default to 0, i.e. the free tier."""
    with _lock:
        totals = dict(_totals)
    price_in = float(os.environ.get("GEMINI_PRICE_PER_M_INPUT", 0) or 0)
    price_out = float(os.environ.get("GEMINI_PRICE_PER_M_OUTPUT", 0) or 0)
    totals["estimated_cost_usd"] = (
        totals["prompt_tokens"] * price_in + totals["output_tokens"] * price_out
    ) / 1_000_000
    return totals


def reset() -> None:
    with _lock:
        for key in _totals:
            _totals[key] = 0


def delta(before: dict, after: dict) -> dict:
    """Usage between two snapshots - what one run/request consumed."""
    return {key: after[key] - before[key] for key in after}
