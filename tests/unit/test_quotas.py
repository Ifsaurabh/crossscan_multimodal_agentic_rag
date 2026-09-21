import pytest

import quotas
from fake_db import FakeConn

USAGE_ZERO = {"requests": 0, "prompt_tokens": 0, "output_tokens": 0}


def make_user(role="user", request_limit=None, token_limit=None):
    return {
        "user_id": "u1", "username": "alice", "role": role,
        "daily_request_limit": request_limit, "daily_token_limit": token_limit,
    }


def clean_env(monkeypatch):
    for name in (
        "USER_DAILY_REQUEST_LIMIT", "USER_DAILY_TOKEN_LIMIT", "ADMIN_DAILY_REQUEST_LIMIT",
        "ADMIN_DAILY_TOKEN_LIMIT", "RATE_LIMIT_PER_MINUTE", "GLOBAL_DAILY_REQUEST_CAP",
        "DAILY_ACTIVE_USER_CAP", "MAX_CONCURRENT_REQUESTS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_evaluate_allows_within_limits():
    assert quotas.evaluate(USAGE_ZERO, (10, 1000), 0, 0) is None


def test_evaluate_daily_request_limit():
    usage = {"requests": 10, "prompt_tokens": 0, "output_tokens": 0}
    assert quotas.evaluate(usage, (10, 1000), 0, 0) == "daily_request_limit"


def test_evaluate_daily_token_limit_counts_prompt_and_output():
    usage = {"requests": 1, "prompt_tokens": 600, "output_tokens": 400}
    assert quotas.evaluate(usage, (10, 1000), 0, 0) == "daily_token_limit"


def test_evaluate_global_cap_only_when_set():
    assert quotas.evaluate(USAGE_ZERO, (10, 1000), 999, 0) is None
    assert quotas.evaluate(USAGE_ZERO, (10, 1000), 5, 5) == "global_daily_cap"


def test_evaluate_none_limit_means_unlimited():
    usage = {"requests": 10**6, "prompt_tokens": 10**9, "output_tokens": 0}
    assert quotas.evaluate(usage, (None, None), 0, 0) is None


def test_limits_for_uses_role_defaults_then_user_overrides(monkeypatch):
    clean_env(monkeypatch)
    assert quotas.limits_for(make_user("user")) == (2, 200_000)  # Decision 2: 2 messages/day for users
    assert quotas.limits_for(make_user("admin")) == (500, 5_000_000)
    assert quotas.limits_for(make_user("user", request_limit=5)) == (5, 200_000)
    assert quotas.limits_for(make_user("user", request_limit=0, token_limit=7)) == (0, 7)


def test_env_overrides_role_defaults(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("USER_DAILY_REQUEST_LIMIT", "3")
    monkeypatch.setenv("ADMIN_DAILY_TOKEN_LIMIT", "99")

    assert quotas.default_limits("user")[0] == 3
    assert quotas.default_limits("admin")[1] == 99


def test_rate_limiter_allows_up_to_limit_then_refuses_with_retry_after():
    limiter = quotas.RateLimiter(limit=2, window_seconds=60)

    assert limiter.allow("k", now=0.0) == (True, 0)
    assert limiter.allow("k", now=1.0) == (True, 0)
    allowed, retry_after = limiter.allow("k", now=2.0)

    assert allowed is False
    assert 0 < retry_after <= 60


def test_rate_limiter_window_slides_and_keys_are_independent():
    limiter = quotas.RateLimiter(limit=1, window_seconds=60)
    limiter.allow("a", now=0.0)

    assert limiter.allow("a", now=30.0)[0] is False
    assert limiter.allow("b", now=30.0)[0] is True
    assert limiter.allow("a", now=61.0)[0] is True


def test_refused_attempts_do_not_consume_rate_slots():
    limiter = quotas.RateLimiter(limit=1, window_seconds=60)
    limiter.allow("a", now=0.0)
    for t in (1.0, 2.0, 3.0):
        limiter.allow("a", now=t)

    assert limiter.allow("a", now=60.5)[0] is True


def test_rate_limiter_reads_limit_from_env_when_not_given(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    limiter = quotas.RateLimiter()
    assert limiter.allow("k", now=0.0)[0] is True
    assert limiter.allow("k", now=1.0)[0] is False


def _quota_conn(requests=0, prompt=0, output=0, global_requests=0, active_users=0):
    return FakeConn(responses=[
        ("COUNT(*)", [(active_users,)]),
        ("SUM(requests)", [(global_requests,)]),
        ("usage_daily", [(requests, prompt, output)]),
    ])


def test_check_quota_passes_for_a_fresh_user(monkeypatch):
    clean_env(monkeypatch)
    quotas.check_quota(_quota_conn(), make_user(), limiter=quotas.RateLimiter(limit=5))


def test_check_quota_refuses_over_daily_limit_without_using_a_rate_slot(monkeypatch):
    clean_env(monkeypatch)
    limiter = quotas.RateLimiter(limit=1)

    with pytest.raises(quotas.QuotaExceeded) as excinfo:
        quotas.check_quota(_quota_conn(requests=2), make_user(), limiter=limiter)

    assert excinfo.value.reason == "daily_request_limit"
    assert limiter.allow("u1")[0] is True  # slot was not consumed by the refusal


def test_check_quota_rate_limit_carries_retry_after(monkeypatch):
    clean_env(monkeypatch)
    limiter = quotas.RateLimiter(limit=1)
    quotas.check_quota(_quota_conn(), make_user(), limiter=limiter)

    with pytest.raises(quotas.QuotaExceeded) as excinfo:
        quotas.check_quota(_quota_conn(), make_user(), limiter=limiter)

    assert excinfo.value.reason == "rate_limit"
    assert excinfo.value.retry_after > 0


def test_check_quota_global_cap(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("GLOBAL_DAILY_REQUEST_CAP", "5")

    with pytest.raises(quotas.QuotaExceeded) as excinfo:
        quotas.check_quota(_quota_conn(global_requests=5), make_user(), limiter=quotas.RateLimiter(limit=5))

    assert excinfo.value.reason == "global_daily_cap"


def test_daily_user_cap_defaults_and_is_env_overridable(monkeypatch):
    clean_env(monkeypatch)
    assert quotas.daily_user_cap() == 20
    monkeypatch.setenv("DAILY_ACTIVE_USER_CAP", "5")
    assert quotas.daily_user_cap() == 5


def test_a_new_user_is_refused_once_the_daily_user_cap_is_full(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("DAILY_ACTIVE_USER_CAP", "3")

    with pytest.raises(quotas.QuotaExceeded) as excinfo:
        quotas.check_quota(_quota_conn(requests=0, active_users=3), make_user(), limiter=quotas.RateLimiter(limit=5))

    assert excinfo.value.reason == "daily_user_cap"


def test_a_new_user_is_admitted_while_the_cap_has_room(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("DAILY_ACTIVE_USER_CAP", "3")

    quotas.check_quota(_quota_conn(requests=0, active_users=2), make_user(), limiter=quotas.RateLimiter(limit=5))


def test_a_user_already_served_today_is_never_cut_off_by_the_user_cap(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("DAILY_ACTIVE_USER_CAP", "3")
    conn = _quota_conn(requests=1, active_users=99)

    quotas.check_quota(conn, make_user(), limiter=quotas.RateLimiter(limit=5))

    assert conn.find("COUNT(*)") == []  # the check does not even run for them


def test_admins_are_exempt_from_the_daily_user_cap(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("DAILY_ACTIVE_USER_CAP", "1")

    quotas.check_quota(_quota_conn(requests=0, active_users=50), make_user("admin"), limiter=quotas.RateLimiter(limit=5))


def test_daily_user_cap_zero_disables_it(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("DAILY_ACTIVE_USER_CAP", "0")
    conn = _quota_conn(requests=0, active_users=999)

    quotas.check_quota(conn, make_user(), limiter=quotas.RateLimiter(limit=5))

    assert conn.find("COUNT(*)") == []


def test_concurrency_limiter_refuses_when_every_slot_is_taken():
    limiter = quotas.ConcurrencyLimiter(limit=2)

    with limiter.slot():
        with limiter.slot():
            assert limiter.active == 2
            with pytest.raises(quotas.QuotaExceeded) as excinfo:
                with limiter.slot():
                    pytest.fail("a third request must not get a slot")
            assert excinfo.value.reason == "busy" and excinfo.value.retry_after == 10

    assert limiter.active == 0


def test_concurrency_slot_is_released_even_when_the_work_fails():
    limiter = quotas.ConcurrencyLimiter(limit=1)

    with pytest.raises(RuntimeError):
        with limiter.slot():
            raise RuntimeError("boom")

    assert limiter.active == 0
    with limiter.slot():  # the slot is available again
        pass


def test_a_refused_request_does_not_take_or_leak_a_slot():
    limiter = quotas.ConcurrencyLimiter(limit=1)
    assert limiter.acquire() is True

    assert limiter.acquire() is False
    limiter.release()

    assert limiter.active == 0
    limiter.release()  # releasing too often never goes negative
    assert limiter.active == 0


def test_concurrency_limit_defaults_to_three_and_zero_disables(monkeypatch):
    clean_env(monkeypatch)
    assert quotas.max_concurrent_requests() == 3
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "0")
    limiter = quotas.ConcurrencyLimiter()

    assert all(limiter.acquire() for _ in range(50))


def test_concurrency_limiter_holds_up_under_real_threads():
    import threading

    limiter = quotas.ConcurrencyLimiter(limit=3)
    gate, entered, refused = threading.Event(), [], []

    def worker():
        try:
            with limiter.slot():
                entered.append(1)
                gate.wait(timeout=5)
        except quotas.QuotaExceeded:
            refused.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    while len(entered) + len(refused) < 8:
        threading.Event().wait(0.01)
    gate.set()
    for t in threads:
        t.join()

    assert len(entered) == 3 and len(refused) == 5
    assert limiter.active == 0


def test_global_cap_is_on_by_default_and_env_overridable(monkeypatch):
    clean_env(monkeypatch)
    assert quotas.global_daily_cap() == quotas.DEFAULT_GLOBAL_DAILY_CAP == 50

    monkeypatch.setenv("GLOBAL_DAILY_REQUEST_CAP", "200")
    assert quotas.global_daily_cap() == 200


def test_default_global_cap_blocks_when_reached(monkeypatch):
    clean_env(monkeypatch)

    with pytest.raises(quotas.QuotaExceeded) as excinfo:
        quotas.check_quota(_quota_conn(global_requests=50), make_user(), limiter=quotas.RateLimiter(limit=5))

    assert excinfo.value.reason == "global_daily_cap"


def test_global_total_is_not_queried_when_cap_is_explicitly_disabled(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("GLOBAL_DAILY_REQUEST_CAP", "0")
    conn = _quota_conn()

    quotas.check_quota(conn, make_user(), limiter=quotas.RateLimiter(limit=5))

    assert conn.find("SUM(requests)") == []


def test_a_regular_user_gets_a_first_question_and_one_follow_up_per_day(monkeypatch):
    clean_env(monkeypatch)
    user = make_user()

    quotas.check_quota(_quota_conn(requests=0), user, limiter=quotas.RateLimiter(limit=5))  # 1st question
    quotas.check_quota(_quota_conn(requests=1), user, limiter=quotas.RateLimiter(limit=5))  # follow-up
    with pytest.raises(quotas.QuotaExceeded) as excinfo:
        quotas.check_quota(_quota_conn(requests=2), user, limiter=quotas.RateLimiter(limit=5))  # third

    assert excinfo.value.reason == "daily_request_limit"


def test_admins_are_not_held_to_the_two_message_limit(monkeypatch):
    clean_env(monkeypatch)
    quotas.check_quota(_quota_conn(requests=50), make_user("admin"), limiter=quotas.RateLimiter(limit=5))


def test_registration_is_throttled_service_wide():
    limiter = quotas.RateLimiter(limit=2, window_seconds=3600)

    quotas.check_registration_allowed(limiter)
    quotas.check_registration_allowed(limiter)
    with pytest.raises(quotas.QuotaExceeded) as excinfo:
        quotas.check_registration_allowed(limiter)

    assert excinfo.value.reason == "registration_rate_limit"
    assert excinfo.value.retry_after > 0
    assert "accounts" in str(excinfo.value)


def test_registration_limit_default_is_ten_per_hour():
    assert quotas.DEFAULT_REGISTRATIONS_PER_HOUR == 10


def test_quota_exceeded_message_is_user_friendly():
    assert "daily message limit" in str(quotas.QuotaExceeded("daily_request_limit"))


def test_record_usage_upserts_and_commits():
    conn = FakeConn()

    quotas.record_usage(conn, "u1", 120, 30)

    sql, params = conn.find("INSERT INTO")[0]
    assert "ON CONFLICT" in sql
    assert params == ("u1", 120, 30)
    assert conn.commits == 1


def test_remaining_reports_used_limit_and_left(monkeypatch):
    clean_env(monkeypatch)
    conn = _quota_conn(requests=1, prompt=1000, output=500)

    result = quotas.remaining(conn, make_user())

    assert result["requests_used"] == 1
    assert result["requests_left"] == 1
    assert result["tokens_used"] == 1500
    assert result["tokens_left"] == 200_000 - 1500


def test_remaining_never_goes_negative(monkeypatch):
    clean_env(monkeypatch)
    conn = _quota_conn(requests=99)
    assert quotas.remaining(conn, make_user())["requests_left"] == 0
