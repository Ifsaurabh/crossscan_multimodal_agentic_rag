"""parallel: independent searches at the same time, results in the order given."""
import threading
import time

import pytest

from retrieval import parallel, tracing
from shared import langfuse_client


def test_the_results_come_back_in_the_order_the_tasks_were_given():
    def task(i, wait):
        def run():
            time.sleep(wait)
            return i
        return run

    assert parallel.run_ordered([task(0, 0.15), task(1, 0.0), task(2, 0.05)]) == [0, 1, 2]


def test_the_tasks_really_overlap():
    barrier = threading.Barrier(3, timeout=5)  # passes only if all three are running at once
    assert parallel.run_ordered([lambda: barrier.wait(), lambda: barrier.wait(), lambda: barrier.wait()]) is not None


def test_one_task_or_none_runs_in_the_calling_thread():
    caller = threading.current_thread()
    assert parallel.run_ordered([lambda: threading.current_thread()]) == [caller]
    assert parallel.run_ordered([]) == []


def test_with_the_switch_off_everything_runs_in_the_calling_thread_in_order(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_PARALLEL", "0")
    caller, order = threading.current_thread(), []
    results = parallel.run_ordered([lambda i=i: order.append(i) or threading.current_thread() for i in range(3)])
    assert results == [caller] * 3 and order == [0, 1, 2]


def test_the_error_of_the_first_failing_task_is_raised_after_all_have_finished():
    finished = []

    def failing(name, wait):
        def run():
            time.sleep(wait)
            finished.append(name)
            raise ValueError(name)
        return run

    with pytest.raises(ValueError, match="first"):
        parallel.run_ordered([failing("first", 0.1), failing("second", 0.0)])

    assert sorted(finished) == ["first", "second"]


def test_the_timings_of_the_question_are_added_to_from_every_thread():
    tracing.reset()

    def task():
        with tracing.timed("db_s"):
            time.sleep(0.05)

    parallel.run_ordered([task, task, task])

    assert tracing.snapshot()["db_s"] >= 0.14  # summed over the three threads, not the wall time


def test_a_span_opened_in_a_task_is_still_a_child_of_the_step_that_started_it(monkeypatch):
    stack_seen = {}

    class Obs:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def update(self, **k):
            pass

    import contextvars

    current = contextvars.ContextVar("current_span", default=None)

    class Client:
        def start_as_current_observation(self, **kwargs):
            parent = current.get()
            stack_seen[kwargs["name"]] = parent
            token = current.set(kwargs["name"])

            class CM(Obs):
                def __exit__(self, *a):
                    current.reset(token)
                    return False

            return CM(kwargs["name"])

    monkeypatch.setattr(langfuse_client, "get_client", lambda: Client())

    def task():
        with langfuse_client.span("search_vector"):
            pass

    with langfuse_client.span("retrieval_executor"):
        parallel.run_ordered([task, task])

    assert stack_seen["search_vector"] == "retrieval_executor"
