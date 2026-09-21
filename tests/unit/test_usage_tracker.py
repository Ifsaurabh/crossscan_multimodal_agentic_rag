import usage_tracker as ut


class Meta:
    prompt_token_count = 100
    candidates_token_count = 20
    cached_content_token_count = 5


class Resp:
    usage_metadata = Meta()


def setup_function():
    ut.reset()


def test_record_accumulates_tokens_and_calls():
    ut.record(Resp())
    ut.record(Resp())

    snap = ut.snapshot()
    assert snap["calls"] == 2
    assert snap["prompt_tokens"] == 200
    assert snap["output_tokens"] == 40
    assert snap["cached_tokens"] == 10


def test_record_tokens_is_provider_neutral_and_feeds_request_scope():
    ut.begin_request()
    ut.record_tokens(50, 10, 3)
    usage = ut.end_request()

    assert usage == {"calls": 1, "prompt_tokens": 50, "output_tokens": 10, "cached_tokens": 3}
    snap = ut.snapshot()
    assert (snap["calls"], snap["prompt_tokens"], snap["output_tokens"], snap["cached_tokens"]) == (1, 50, 10, 3)


def test_record_tolerates_responses_without_usage():
    ut.record("plain string response")
    ut.record(object())

    snap = ut.snapshot()
    assert snap["calls"] == 2
    assert snap["prompt_tokens"] == 0


def test_cost_defaults_to_zero_free_tier(monkeypatch):
    monkeypatch.delenv("GEMINI_PRICE_PER_M_INPUT", raising=False)
    monkeypatch.delenv("GEMINI_PRICE_PER_M_OUTPUT", raising=False)
    ut.record(Resp())

    assert ut.snapshot()["estimated_cost_usd"] == 0


def test_cost_uses_configured_prices(monkeypatch):
    monkeypatch.setenv("GEMINI_PRICE_PER_M_INPUT", "1.0")
    monkeypatch.setenv("GEMINI_PRICE_PER_M_OUTPUT", "4.0")
    for _ in range(10_000):
        ut.record(Resp())

    # 1,000,000 prompt tokens * $1/M + 200,000 output tokens * $4/M
    assert abs(ut.snapshot()["estimated_cost_usd"] - 1.8) < 1e-9


def test_request_scope_counts_only_usage_recorded_inside_it():
    ut.record(Resp())  # before the request: only in the global totals

    ut.begin_request()
    ut.record(Resp())
    ut.record(Resp())
    usage = ut.end_request()

    assert usage["calls"] == 2
    assert usage["prompt_tokens"] == 200
    assert usage["output_tokens"] == 40
    assert ut.snapshot()["calls"] == 3  # global totals still include everything


def test_end_request_without_begin_returns_zeros_and_recording_is_safe():
    assert ut.end_request() == {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    ut.record(Resp())  # no active request: must not raise


def test_request_scopes_do_not_leak_between_requests():
    ut.begin_request()
    ut.record(Resp())
    ut.end_request()

    ut.begin_request()
    usage = ut.end_request()

    assert usage["calls"] == 0


def test_request_scope_is_isolated_across_threads():
    import threading

    results = {}

    def worker(name, n):
        ut.begin_request()
        for _ in range(n):
            ut.record(Resp())
        results[name] = ut.end_request()["calls"]

    threads = [threading.Thread(target=worker, args=("a", 3)), threading.Thread(target=worker, args=("b", 5))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == {"a": 3, "b": 5}


def test_delta_and_reset():
    before = ut.snapshot()
    ut.record(Resp())
    diff = ut.delta(before, ut.snapshot())

    assert diff["calls"] == 1
    assert diff["prompt_tokens"] == 100

    ut.reset()
    assert ut.snapshot()["calls"] == 0
