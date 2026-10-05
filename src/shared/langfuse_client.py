import contextlib
import os
import sys

from dotenv import load_dotenv

load_dotenv()

_client = None
_client_failed = False


def is_enabled() -> bool:
    """Langfuse is opt-out: on when keys are configured, off if
    LANGFUSE_DISABLED is set (tests) or keys are missing."""
    if os.environ.get("LANGFUSE_DISABLED"):
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def get_client():
    """Shared Langfuse client, or None if disabled/unavailable. Never raises -
    observability must not be able to break the pipeline."""
    global _client, _client_failed
    if not is_enabled() or _client_failed:
        return None
    if _client is None:
        try:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
                timeout=5,
            )
        except Exception as e:
            print(f"   (Langfuse unavailable, continuing without it: {e})")
            _client_failed = True
            return None
    return _client


def get_callbacks() -> list:
    """LangChain/LangGraph callback handlers that trace a graph run to
    Langfuse. Empty list when Langfuse is off, so callers can always pass
    the result straight into graph.invoke(config={"callbacks": ...})."""
    if get_client() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception as e:
        print(f"   (Langfuse tracing unavailable, continuing without it: {e})")
        return []


def flush():
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass


def score_current_trace(name: str, value: float, comment: str = None) -> None:
    """Attach a numeric score to the trace currently running (inside a graph
    node). Never raises and does nothing when Langfuse is off - same
    "observability must not break the pipeline" rule as the rest of this file."""
    client = get_client()
    if client is None:
        return
    try:
        client.score_current_trace(name=name, value=float(value), data_type="NUMERIC", comment=comment)
    except Exception as e:
        print(f"   (Langfuse score '{name}' not recorded: {e})")


class _NoOpGeneration:
    """Stand-in used when Langfuse is off or logging fails - callers never
    need to check, they just call .set_usage() unconditionally."""

    def set_usage(self, *args, **kwargs):
        pass


class _RealGeneration:
    def __init__(self, span):
        self._span = span

    def set_usage(self, output_text: str, prompt_tokens: int, output_tokens: int, cached_tokens: int = 0):
        try:
            self._span.update(
                output=output_text,
                usage_details={
                    "input": prompt_tokens,
                    "output": output_tokens,
                    "cache_read_input_tokens": cached_tokens,
                },
            )
        except Exception:
            pass


@contextlib.contextmanager
def generation_span(name: str, model: str, input_text: str):
    """Log one LLM call to Langfuse as a proper 'generation' observation
    (model name + token usage attached), NOT just the generic LangGraph node
    span the CallbackHandler already captures.

    Why this exists: llm_connection.py calls the Gemini SDK directly rather
    than through a LangChain chat model, so Langfuse's CallbackHandler has no
    visibility into the call as a "generation" - it only sees the LangGraph
    node it ran inside. Checked directly against the Langfuse API: every
    observation from a live run was type CHAIN, none was type GENERATION, so
    no model/token data was ever reaching Langfuse. This wraps the call
    explicitly so token usage shows up, same "never break the pipeline"
    fallback pattern as get_client()/get_callbacks(): yields a no-op when
    Langfuse is off or logging itself fails, so a caller never has to check.

    IMPORTANT (found by live-testing this fix, not by inspection): a naive
    `try: yield ... except Exception: yield NoOp()` around the whole thing is
    wrong - if the CALLER's code raises (e.g. the Gemini call itself fails
    with a real quota error), that exception resumes at the `yield` and lands
    in this except clause too, gets misreported as a Langfuse problem, and
    the second `yield` after it violates the generator-contextmanager
    protocol - which replaces the original exception with a generic
    RuntimeError before it ever reaches llm_connection.generate()'s error
    classifier. Confirmed live: a genuine 429 quota error got misclassified
    as "other" instead of "quota", which would have given it the wrong (0s
    instead of 900s) cooldown - a real regression in the Story-2 failover
    design. Fix: only the Langfuse SETUP (entering the span) is inside a
    try/except; the caller's code runs outside it, with the underlying
    context manager's __enter__/__exit__ driven manually so a caller
    exception still propagates through Langfuse's own exit (so the span
    correctly records the error) and then reaches llm_connection.generate()
    completely unchanged."""
    client = get_client()
    if client is None:
        yield _NoOpGeneration()
        return

    try:
        span_cm = client.start_as_current_observation(
            name=name, as_type="generation", model=model, input=input_text,
        )
        span = span_cm.__enter__()
    except Exception as e:
        print(f"   (Langfuse generation logging unavailable, continuing without it: {e})")
        yield _NoOpGeneration()
        return

    try:
        yield _RealGeneration(span)
    except BaseException:
        span_cm.__exit__(*sys.exc_info())
        raise
    else:
        span_cm.__exit__(None, None, None)


class _NoOpSpan:
    """Stand-in used when Langfuse is off or logging fails: callers just call .update() and never check."""

    def update(self, **kwargs):
        pass


class _RealSpan:
    def __init__(self, span):
        self._span = span

    def update(self, **kwargs):
        try:
            self._span.update(**kwargs)
        except Exception:
            pass


@contextlib.contextmanager
def span(name: str, input=None, metadata: dict = None):
    """One step of the pipeline as a Langfuse span, a child of whatever step is running (a graph node, another span).
    The one helper for every step, so a new step is one `with` line and cannot be forgotten:

        with langfuse_client.span("rerank", metadata={"passages": 12}) as step:
            ...
            step.update(metadata={"rows": 5})        # numbers learned while the step ran

    Never put personal data in `input` or `metadata`: use counts, flags and scores, or text that is already redacted.
    Does nothing, and never raises, when Langfuse is off (LANGFUSE_DISABLED=1, as in CI) or logging itself fails.
    Built like generation_span, and for the same reason: only entering the span is inside the try. The caller's code
    runs outside it, so an exception of the caller still reaches the caller unchanged (and the span records the error)."""
    client = get_client()
    if client is None:
        yield _NoOpSpan()
        return

    try:
        kwargs = {"name": name, "as_type": "span"}
        if input is not None:
            kwargs["input"] = input
        if metadata:
            kwargs["metadata"] = metadata
        span_cm = client.start_as_current_observation(**kwargs)
        step = span_cm.__enter__()
    except Exception as e:
        print(f"   (Langfuse span '{name}' unavailable, continuing without it: {e})")
        yield _NoOpSpan()
        return

    try:
        yield _RealSpan(step)
    except BaseException:
        span_cm.__exit__(*sys.exc_info())
        raise
    else:
        span_cm.__exit__(None, None, None)
