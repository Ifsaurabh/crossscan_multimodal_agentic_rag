import os
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

from db import SCHEMA_NAME

# (daily requests, daily tokens) per role. Overridable per user in the
# users table, and globally via environment variables.
_ROLE_DEFAULTS = {
    # Decision 2 (user): open sign-up, each user gets 2 messages a day for now
    # (a first question and one follow-up). Raise USER_DAILY_REQUEST_LIMIT later.
    "user": (2, 200_000),
    "admin": (500, 5_000_000),
}

QUOTA_MESSAGES = {
    "registration_rate_limit": "Too many accounts were created recently. Please try again later.",
    "rate_limit": "You are sending messages too fast. Please wait a moment.",
    "daily_request_limit": "You have reached your daily message limit. It resets at midnight.",
    "daily_token_limit": "You have reached your daily usage limit. It resets at midnight.",
    "global_daily_cap": "The service has reached its daily capacity. Please try again tomorrow.",
    "daily_user_cap": "The service has reached the number of users it can serve today. Please try again tomorrow.",
    "busy": "The service is busy answering other users right now. Please try again in a few seconds.",
}


class QuotaExceeded(Exception):
    def __init__(self, reason: str, retry_after: int = None):
        super().__init__(QUOTA_MESSAGES.get(reason, reason))
        self.reason = reason
        self.retry_after = retry_after


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def default_limits(role: str) -> tuple:
    request_default, token_default = _ROLE_DEFAULTS.get(role, _ROLE_DEFAULTS["user"])
    prefix = "ADMIN" if role == "admin" else "USER"
    return (
        _env_int(f"{prefix}_DAILY_REQUEST_LIMIT", request_default),
        _env_int(f"{prefix}_DAILY_TOKEN_LIMIT", token_default),
    )


def limits_for(user: dict) -> tuple:
    """A user's own limits win; NULL columns fall back to the role default."""
    request_default, token_default = default_limits(user.get("role", "user"))
    request_limit = user.get("daily_request_limit")
    token_limit = user.get("daily_token_limit")
    return (
        request_default if request_limit is None else request_limit,
        token_default if token_limit is None else token_limit,
    )


DEFAULT_GLOBAL_DAILY_CAP = 50


def global_daily_cap() -> int:
    """Total requests per day across ALL users. Always on by default (a
    limit is mandatory); raise GLOBAL_DAILY_REQUEST_CAP when the upstream
    quota grows (paid Gemini, another provider). One request is roughly 3-4
    model calls, so Gemini's free tier of 20 calls/day only supports ~5.
    Setting it to 0 explicitly disables the cap."""
    return _env_int("GLOBAL_DAILY_REQUEST_CAP", DEFAULT_GLOBAL_DAILY_CAP)


DEFAULT_DAILY_USER_CAP = 20
DEFAULT_MAX_CONCURRENT = 3


def daily_user_cap() -> int:
    """How many DIFFERENT users may be served per day. Someone already served
    today is never cut off by this (their own limits still apply); only a
    new user is refused once the cap is reached. Admins are exempt. With the
    default 2 messages per user, 20 users = 40 requests, inside the global
    cap of 50. 0 disables."""
    return _env_int("DAILY_ACTIVE_USER_CAP", DEFAULT_DAILY_USER_CAP)


def max_concurrent_requests() -> int:
    """How many questions may be processed at the SAME moment (each one is
    several model calls plus retrieval). 0 disables."""
    return _env_int("MAX_CONCURRENT_REQUESTS", DEFAULT_MAX_CONCURRENT)


def evaluate(usage: dict, limits: tuple, global_requests: int, cap: int):
    """Pure decision: the reason a request must be refused, or None."""
    request_limit, token_limit = limits
    if request_limit is not None and usage["requests"] >= request_limit:
        return "daily_request_limit"
    if token_limit is not None and usage["prompt_tokens"] + usage["output_tokens"] >= token_limit:
        return "daily_token_limit"
    if cap and global_requests >= cap:
        return "global_daily_cap"
    return None


class RateLimiter:
    """Sliding-window limiter, per key, in memory (one process). A hard
    per-minute ceiling on top of the daily quota."""

    def __init__(self, limit: int = None, window_seconds: float = 60.0):
        self._limit = limit
        self._window = window_seconds
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def limit(self) -> int:
        return self._limit if self._limit is not None else _env_int("RATE_LIMIT_PER_MINUTE", 10)

    def allow(self, key: str, now: float = None):
        """Returns (allowed, retry_after_seconds). A refused attempt does not
        consume a slot."""
        now = time.monotonic() if now is None else now
        with self._lock:
            events = self._events[key]
            while events and now - events[0] >= self._window:
                events.popleft()
            if len(events) >= self.limit():
                return False, int(self._window - (now - events[0])) + 1
            events.append(now)
            return True, 0


rate_limiter = RateLimiter()


class ConcurrencyLimiter:
    """Caps how many requests run at once (in memory, one process). Never
    waits: when every slot is taken the caller is refused immediately with
    'busy', which is friendlier and safer than queueing threads."""

    def __init__(self, limit: int = None):
        self._limit = limit
        self._active = 0
        self._lock = threading.Lock()

    def limit(self) -> int:
        return self._limit if self._limit is not None else max_concurrent_requests()

    @property
    def active(self) -> int:
        return self._active

    def acquire(self) -> bool:
        with self._lock:
            limit = self.limit()
            if limit and self._active >= limit:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    @contextmanager
    def slot(self, retry_after: int = 10):
        """Holds one slot for the `with` body; raises QuotaExceeded('busy')
        when none is free. The slot is always released, even on errors."""
        if not self.acquire():
            raise QuotaExceeded("busy", retry_after=retry_after)
        try:
            yield
        finally:
            self.release()


concurrency_limiter = ConcurrencyLimiter()

# With open sign-up one person could create many accounts to get around the
# per-user daily limit. This caps how many accounts can be created per hour
# (service-wide; there is no per-IP view). The global daily cap is the real
# backstop for the upstream model quota.
DEFAULT_REGISTRATIONS_PER_HOUR = 10
registration_limiter = RateLimiter(
    limit=_env_int("REGISTRATIONS_PER_HOUR", DEFAULT_REGISTRATIONS_PER_HOUR), window_seconds=3600.0,
)


def check_registration_allowed(limiter: RateLimiter = None) -> None:
    """Raises QuotaExceeded('registration_rate_limit') when too many accounts
    were created in the last hour."""
    allowed, retry_after = (limiter or registration_limiter).allow("registration")
    if not allowed:
        raise QuotaExceeded("registration_rate_limit", retry_after=retry_after)


def get_usage_today(conn, user_id: str) -> dict:
    row = conn.execute(
        f"""SELECT requests, prompt_tokens, output_tokens FROM {SCHEMA_NAME}.usage_daily
            WHERE user_id = %s AND day = current_date""",
        (user_id,),
    ).fetchone()
    if row is None:
        return {"requests": 0, "prompt_tokens": 0, "output_tokens": 0}
    return {"requests": row[0], "prompt_tokens": row[1], "output_tokens": row[2]}


def get_global_requests_today(conn) -> int:
    row = conn.execute(
        f"SELECT COALESCE(SUM(requests), 0) FROM {SCHEMA_NAME}.usage_daily WHERE day = current_date"
    ).fetchone()
    return int(row[0])


def get_active_users_today(conn) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM {SCHEMA_NAME}.usage_daily WHERE day = current_date"
    ).fetchone()
    return int(row[0])


def check_quota(conn, user: dict, limiter: RateLimiter = None) -> None:
    """Raises QuotaExceeded if this user may not send another request now.
    Daily limits are checked before the rate limiter so that a refused
    request never uses up a rate-limit slot."""
    limiter = limiter or rate_limiter
    usage = get_usage_today(conn, user["user_id"])
    cap = global_daily_cap()
    global_requests = get_global_requests_today(conn) if cap else 0

    reason = evaluate(usage, limits_for(user), global_requests, cap)
    if reason:
        raise QuotaExceeded(reason)

    # A user with no usage row yet is a NEW user today; refuse only them when
    # the day's user cap is full. Admins are exempt.
    user_cap = daily_user_cap()
    if user_cap and usage["requests"] == 0 and user.get("role") != "admin":
        if get_active_users_today(conn) >= user_cap:
            raise QuotaExceeded("daily_user_cap")

    allowed, retry_after = limiter.allow(user["user_id"])
    if not allowed:
        raise QuotaExceeded("rate_limit", retry_after=retry_after)


def record_usage(conn, user_id: str, prompt_tokens: int, output_tokens: int) -> None:
    conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.usage_daily (user_id, day, requests, prompt_tokens, output_tokens)
            VALUES (%s, current_date, 1, %s, %s)
            ON CONFLICT (user_id, day) DO UPDATE SET
                requests = {SCHEMA_NAME}.usage_daily.requests + 1,
                prompt_tokens = {SCHEMA_NAME}.usage_daily.prompt_tokens + EXCLUDED.prompt_tokens,
                output_tokens = {SCHEMA_NAME}.usage_daily.output_tokens + EXCLUDED.output_tokens""",
        (user_id, prompt_tokens, output_tokens),
    )
    conn.commit()


def remaining(conn, user: dict) -> dict:
    """Used/limit/left for display (sidebar, /me)."""
    usage = get_usage_today(conn, user["user_id"])
    request_limit, token_limit = limits_for(user)
    tokens_used = usage["prompt_tokens"] + usage["output_tokens"]
    return {
        "requests_used": usage["requests"],
        "requests_limit": request_limit,
        "requests_left": max(0, request_limit - usage["requests"]) if request_limit is not None else None,
        "tokens_used": tokens_used,
        "tokens_limit": token_limit,
        "tokens_left": max(0, token_limit - tokens_used) if token_limit is not None else None,
    }
