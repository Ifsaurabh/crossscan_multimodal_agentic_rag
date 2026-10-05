import time

from google.genai.errors import ServerError

from shared import usage_tracker

DEFAULT_MAX_RETRIES = 4

# There is no Gemini prompt cache here any more. Every fixed instruction in the code
# is far below Google's 1,024-token minimum for a cache, so it never took effect. A call is a plain generate call, with
# retry-with-backoff on transient server errors.


def _busy_message(error: ServerError, model: str, wait: int) -> str:
    """Says WHICH error Gemini returned (its real status), not just 'busy' -
    a 503 overload and a 500 internal error need different reactions."""
    code = getattr(error, "code", None)
    status = getattr(error, "status", None)
    return f"   (Gemini {model} returned {code or '5xx'} {status or ''}: retrying in {wait}s...)"


def call_with_retry(client, model: str, contents: str, max_retries: int = DEFAULT_MAX_RETRIES):
    """A plain generate call with retry-with-backoff: Gemini occasionally
    returns transient 503s under high demand.
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
