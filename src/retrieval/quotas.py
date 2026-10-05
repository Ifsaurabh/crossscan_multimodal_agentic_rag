import os
import threading
import time
from contextlib import contextmanager

from retrieval.event_log import MemoryEventLog, PgEventLog
from shared.db import SCHEMA_NAME

# Decision (user, 2026-10-05): every daily counter resets at midnight India time (IST, UTC+5:30, no daylight saving).
IST_OFFSET_SECONDS = 19800
TODAY_SQL = "(now() AT TIME ZONE 'Asia/Kolkata')::date"


def ist_midnight(now: float) -> float:
    """The epoch second of the most recent midnight in India."""
    return (now + IST_OFFSET_SECONDS) // 86400 * 86400 - IST_OFFSET_SECONDS


def seconds_to_ist_midnight(now: float) -> int:
    return int(ist_midnight(now) + 86400 - now) + 1

# (daily requests, daily tokens) per role. Overridable per user in the
# users table, and globally via environment variables.
_ROLE_DEFAULTS = {
    # Decision (user, 2026-10-05): open sign-up, each user gets 6 messages a day for now.
    # Raise USER_DAILY_REQUEST_LIMIT later. Admins have no limits at all
    # (see is_unlimited), so there is no admin entry.
    "user": (6, 200_000),
}


def is_unlimited(user: dict) -> bool:
    return user.get("role") == "admin"

QUOTA_MESSAGES = {
    "registration_rate_limit": "Too many accounts were created today. Please try again tomorrow.",
    "rate_limit": "You are sending messages too fast. Please wait a moment.",
    "daily_request_limit": "You have reached your daily message limit. It resets at midnight, India time.",
    "daily_token_limit": "You have reached your daily usage limit. It resets at midnight, India time.",
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
    request_default, token_default = _ROLE_DEFAULTS["user"]
    return (
        _env_int("USER_DAILY_REQUEST_LIMIT", request_default),
        _env_int("USER_DAILY_TOKEN_LIMIT", token_default),
    )


def limits_for(user: dict) -> tuple:
    """A user's own limits win; NULL columns fall back to the role default.
    Admins have no limits (None, None)."""
    if is_unlimited(user):
        return (None, None)
    request_default, token_default = default_limits(user.get("role", "user"))
    request_limit = user.get("daily_request_limit")
    token_limit = user.get("daily_token_limit")
    return (
        request_default if request_limit is None else request_limit,
        token_default if token_limit is None else token_limit,
    )


DEFAULT_GLOBAL_DAILY_CAP = 100


def global_daily_cap() -> int:
    """Total requests per day across ALL users. Always on by default (a
    limit is mandatory); raise GLOBAL_DAILY_REQUEST_CAP when the upstream
    quota grows (paid Gemini, another provider). One request is roughly 3-4
    model calls, so Gemini's free tier of 20 calls/day only supports ~5.
    Setting it to 0 explicitly disables the cap."""
    return _env_int("GLOBAL_DAILY_REQUEST_CAP", DEFAULT_GLOBAL_DAILY_CAP)


DEFAULT_DAILY_USER_CAP = 10
DEFAULT_MAX_CONCURRENT = 3


def daily_user_cap() -> int:
    """How many DIFFERENT users may be served per day. Someone already served
    today is never cut off by this (their own limits still apply); only a
    new user is refused once the cap is reached. Admins are exempt. With the
    default 6 messages per user, 10 users = 60 requests, inside the global
    cap of 100 (the rest is room for admins). 0 disables."""
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


DEFAULT_RATE_LIMIT_PER_MINUTE = 2


class RateLimiter:
    """Sliding-window limiter, per key. The events are kept in `log`: Postgres for the app (shared by every instance),
    memory for tests. `daily=True` makes the window "since midnight India time" instead of a fixed number of seconds.
    A refused attempt does not consume a slot."""

    def __init__(self, limit: int = None, window_seconds: float = 60.0, log=None, bucket: str = "rate", daily: bool = False):
        self._limit = limit
        self._window = window_seconds
        self._log = log if log is not None else MemoryEventLog()
        self._bucket = bucket
        self._daily = daily

    def limit(self) -> int:
        return self._limit if self._limit is not None else _env_int("RATE_LIMIT_PER_MINUTE", DEFAULT_RATE_LIMIT_PER_MINUTE)

    def _since(self, now: float) -> float:
        return ist_midnight(now) if self._daily else now - self._window

    def allow(self, key: str, now: float = None, conn=None):
        """Returns (allowed, retry_after_seconds)."""
        now = time.time() if now is None else now
        allowed, oldest = self._log.try_add(self._bucket, key, self.limit(), self._since(now), now, conn)
        if allowed:
            return True, 0
        if self._daily:
            return False, seconds_to_ist_midnight(now)
        return False, int(self._window - (now - oldest)) + 1

    def refund(self, key: str, conn=None) -> None:
        """Gives back the slot the last allowed attempt took."""
        self._log.remove_latest(self._bucket, key, conn)


rate_limiter = RateLimiter(log=PgEventLog())


class ConcurrencyLimiter:
    """Caps how many requests run at once (in memory, one process). Never
    waits: when every slot is taken the caller is refused immediately with
    'busy', which is friendlier and safer than queueing threads. It stays in memory on purpose: the database pool it
    protects belongs to one process, so the cap is per instance (each instance may run this many questions)."""

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
# per-user daily limit. This caps how many accounts can be created per day
# (service-wide, since midnight India time; there is no per-IP view). The global daily cap is the real
# backstop for the upstream model quota.
DEFAULT_REGISTRATIONS_PER_DAY = 10
registration_limiter = RateLimiter(
    limit=_env_int("REGISTRATIONS_PER_DAY", DEFAULT_REGISTRATIONS_PER_DAY), window_seconds=86400.0,
    log=PgEventLog(), bucket="registration", daily=True,
)


def check_registration_allowed(limiter: RateLimiter = None) -> None:
    """Raises QuotaExceeded('registration_rate_limit') when too many accounts
    were created today."""
    allowed, retry_after = (limiter or registration_limiter).allow("registration")
    if not allowed:
        raise QuotaExceeded("registration_rate_limit", retry_after=retry_after)


def get_usage_today(conn, user_id: str) -> dict:
    row = conn.execute(
        f"""SELECT requests, prompt_tokens, output_tokens FROM {SCHEMA_NAME}.usage_daily
            WHERE user_id = %s AND day = {TODAY_SQL}""",
        (user_id,),
    ).fetchone()
    if row is None:
        return {"requests": 0, "prompt_tokens": 0, "output_tokens": 0}
    return {"requests": row[0], "prompt_tokens": row[1], "output_tokens": row[2]}


def get_global_requests_today(conn) -> int:
    row = conn.execute(
        f"SELECT COALESCE(SUM(requests), 0) FROM {SCHEMA_NAME}.usage_daily WHERE day = {TODAY_SQL}"
    ).fetchone()
    return int(row[0])


def get_active_users_today(conn) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM {SCHEMA_NAME}.usage_daily WHERE day = {TODAY_SQL}"
    ).fetchone()
    return int(row[0])


def check_quota(conn, user: dict, limiter: RateLimiter = None) -> None:
    """Raises QuotaExceeded if this user may not send another request now.
    Daily limits are checked before the rate limiter so that a refused
    request never uses up a rate-limit slot. Admins skip every check."""
    if is_unlimited(user):
        return
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

    allowed, retry_after = limiter.allow(user["user_id"], conn=conn)
    if not allowed:
        raise QuotaExceeded("rate_limit", retry_after=retry_after)


def refund_rate_slot(conn, user: dict, limiter: RateLimiter = None) -> None:
    """A question refused as 'busy' was never answered, so the per-minute slot it took goes back to the user."""
    if not is_unlimited(user):
        (limiter or rate_limiter).refund(user["user_id"], conn=conn)


def record_usage(conn, user_id: str, prompt_tokens: int, output_tokens: int) -> None:
    conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.usage_daily (user_id, day, requests, prompt_tokens, output_tokens)
            VALUES (%s, {TODAY_SQL}, 1, %s, %s)
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
