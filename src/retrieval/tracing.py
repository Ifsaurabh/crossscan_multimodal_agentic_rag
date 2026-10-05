"""tracing: the two things every retrieval step needs, in one helper.

    with tracing.step("rerank", timing="rerank_s", passages=12) as step:
        ...
        step.update(metadata={"scored": 12})

`step` is a Langfuse span (shared/langfuse_client.span: nothing happens when Langfuse is off) and, when `timing` is
given, its seconds are ADDED (summed over parallel searches, so `db_s` can exceed the wall time) to a per-question accumulator under that name (`db_s`, `embed_s`, `rerank_s`,
`retrieval_s`). The accumulator is reset at the start of a question (run_query.invoke_graph) and read at the end, so
the evaluation can log them to MLflow next to `latency_s`, and runs can be compared (section 6).

RULE for the future: any new pipeline step gets a span in the same change, through this helper.
No personal data in spans: only counts, flags, scores and text that is already redacted.
"""
import contextlib
import contextvars
import threading
import time

from shared import langfuse_client

_timings = contextvars.ContextVar("retrieval_timings", default=None)
_lock = threading.Lock()  # searches run in parallel threads and all add to the same accumulator


def reset() -> None:
    """Starts a fresh accumulator (the beginning of a question)."""
    _timings.set({})


def snapshot() -> dict:
    """The seconds accumulated so far, by name."""
    return {name: round(seconds, 3) for name, seconds in (_timings.get() or {}).items()}


def _add(name: str, seconds: float) -> None:
    totals = _timings.get()
    if totals is not None:
        with _lock:
            totals[name] = totals.get(name, 0.0) + seconds


@contextlib.contextmanager
def timed(name: str):
    """Only the timing, no span (for a step that is already a graph node)."""
    started = time.perf_counter()
    try:
        yield
    finally:
        _add(name, time.perf_counter() - started)


@contextlib.contextmanager
def step(name: str, timing: str = None, input=None, **metadata):
    """A span named `name` with `metadata`; `timing` also adds its seconds to the accumulator."""
    started = time.perf_counter()
    with langfuse_client.span(name, input=input, metadata=metadata or None) as span:
        try:
            yield span
        finally:
            if timing:
                _add(timing, time.perf_counter() - started)
