"""parallel: runs independent retrieval searches at the same time, and gives their results back IN THE ORDER THEY WERE GIVEN, so the merged and de-duplicated chunks come out exactly as they did when the searches ran one after another.

Each task runs in a copy of the caller's context, so a Langfuse span opened inside a task is still a child of the step that
started it, and the per-question timings (retrieval/tracing.py) keep adding up. How many database searches run at once is
capped elsewhere, in retrieval_executor (a semaphore sized to the connection pool), so many tasks cannot starve the pool.

RETRIEVAL_PARALLEL=0 turns it off (everything runs one after another): for a before/after measurement, or if it ever misbehaves.
"""
import contextvars
import os
from concurrent.futures import ThreadPoolExecutor

MAX_WORKERS = 8


def enabled() -> bool:
    return os.environ.get("RETRIEVAL_PARALLEL", "1") != "0"


def run_ordered(tasks: list) -> list:
    """Calls every zero-argument function in `tasks` and returns their results in the same order. With one task, or with
    parallelism off, they simply run one after another in the calling thread. If a task raises, the error of the FIRST failing
    task (in order) is raised, after every task has finished."""
    if len(tasks) <= 1 or not enabled():
        return [task() for task in tasks]

    with ThreadPoolExecutor(max_workers=min(len(tasks), MAX_WORKERS), thread_name_prefix="retrieval") as pool:
        futures = [pool.submit(contextvars.copy_context().run, task) for task in tasks]
        return [future.result() for future in futures]
