import time

from google import genai
from google.genai import types
from google.genai.errors import ServerError

import usage_tracker

DEFAULT_MAX_RETRIES = 4

# Gemini refuses to cache less than this many tokens ("Cached content is too
# small"). Estimating from the text length lets us skip a request that is
# certain to fail, instead of discovering it with a real API call.
MIN_CACHEABLE_TOKENS = 1024
CHARS_PER_TOKEN = 4

# Tracks whether caching is available for a given (model, system_instruction)
# pair, so we only ever ATTEMPT cache creation once per distinct instruction
# text, not on every single call. None = known unavailable (fall back to
# plain generation, no wasted retries on every request).
_cache_registry = {}


def _busy_message(error: ServerError, model: str, wait: int) -> str:
    """Says WHICH error Gemini returned (its real status), not just 'busy' -
    a 503 overload and a 500 internal error need different reactions."""
    code = getattr(error, "code", None)
    status = getattr(error, "status", None)
    return f"   (Gemini {model} returned {code or '5xx'} {status or ''}: retrying in {wait}s...)"


def call_with_retry(client, model: str, contents: str, max_retries: int = DEFAULT_MAX_RETRIES):
    """Same retry-with-backoff pattern proven in src/extract_entities.py -
    Gemini occasionally returns transient 503s under high demand.
    `max_retries` is the number of ATTEMPTS (waits in between: 2s, 4s, 8s)."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=contents)
            usage_tracker.record(response)
            return response
        except ServerError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(_busy_message(e, model, wait))
            time.sleep(wait)


def too_small_to_cache(system_instruction: str) -> bool:
    return len(system_instruction) < MIN_CACHEABLE_TOKENS * CHARS_PER_TOKEN


def _get_cached_content_name(client, model: str, system_instruction: str):
    """Attempts to create (or reuse an already-attempted) Gemini context
    cache for a fixed instruction block. Returns the cache name on success,
    or None if caching isn't available (no billing, instruction too short
    for the minimum cacheable size, etc.) - caller must fall back to plain
    generation in that case. Any failure here is caught and remembered so
    we don't retry a doomed cache-create call on every single request."""
    key = (model, system_instruction)
    if key in _cache_registry:
        return _cache_registry[key]

    if too_small_to_cache(system_instruction):
        _cache_registry[key] = None  # certain to be refused: don't spend a request finding out
        return None

    try:
        cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                system_instruction=system_instruction,
                ttl="3600s",
            ),
        )
        _cache_registry[key] = cache.name
    except Exception as e:
        print(f"   (prompt caching unavailable, falling back to normal generation: {e})")
        _cache_registry[key] = None

    return _cache_registry[key]


def call_with_cache(client, model: str, system_instruction: str, contents: str, max_retries: int = DEFAULT_MAX_RETRIES):
    """Opportunistic prompt caching: attempts to cache the fixed
    system_instruction block once per (model, instruction), reusing it on
    every subsequent call. Falls back to a normal (uncached) call with the
    instruction inlined if caching isn't available - callers never need to
    handle the difference, this always returns a normal generate_content
    response either way."""
    cached_name = _get_cached_content_name(client, model, system_instruction)

    for attempt in range(max_retries):
        try:
            if cached_name:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(cached_content=cached_name),
                )
            else:
                response = client.models.generate_content(
                    model=model,
                    contents=f"{system_instruction}\n\n{contents}",
                )
            usage_tracker.record(response)
            return response
        except ServerError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(_busy_message(e, model, wait))
            time.sleep(wait)
