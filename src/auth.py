import hashlib
import hmac
import os
import re
import secrets
import time
import uuid

import psycopg

from db import SCHEMA_NAME

SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
MIN_PASSWORD_LENGTH = 8
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
ROLES = ("user", "admin")

MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 900

# Columns an admin may change via update_user (whitelist - column names are
# interpolated into SQL, values never are).
UPDATABLE_FIELDS = {"role", "is_active", "daily_request_limit", "daily_token_limit"}

_failed_logins = {}  # username -> [timestamps of recent failures]
_dummy_hash = None


class AuthError(Exception):
    """Bad credentials, duplicate username, invalid input."""


class LockedOut(AuthError):
    """Too many failed logins for this username."""

    def __init__(self, retry_after: int):
        super().__init__("Too many failed login attempts. Try again later.")
        self.retry_after = retry_after


class PermissionDenied(Exception):
    """Authenticated, but not allowed to do this."""


def token_ttl_hours() -> int:
    return int(os.environ.get("AUTH_TOKEN_TTL_HOURS", 24))


def registration_enabled() -> bool:
    """Self-service sign-up. ON by default (user Decision 2: open sign-up with
    tight per-user limits); set ALLOW_REGISTRATION=0 for admin-created
    accounts only."""
    return os.environ.get("ALLOW_REGISTRATION", "1") != "0"


def hash_password(password: str) -> str:
    """scrypt (memory-hard, in the standard library) with a random per-user
    salt. Stored as scrypt$N$r$p$salt_hex$hash_hex so parameters can change
    later without breaking existing hashes."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def hash_token(token: str) -> str:
    """Only the SHA-256 of a login token is stored, so a leaked database
    does not leak usable tokens."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _dummy_password_hash() -> str:
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password("dummy-password-for-timing")
    return _dummy_hash


def _prune(username: str, now: float) -> list:
    recent = [t for t in _failed_logins.get(username, []) if now - t < LOCKOUT_SECONDS]
    _failed_logins[username] = recent
    return recent


def check_lockout(username: str, now: float = None) -> None:
    now = time.time() if now is None else now
    recent = _prune(username, now)
    if len(recent) >= MAX_FAILED_LOGINS:
        raise LockedOut(retry_after=int(LOCKOUT_SECONDS - (now - recent[0])) + 1)


def record_failed_login(username: str, now: float = None) -> None:
    now = time.time() if now is None else now
    _prune(username, now).append(now)


def clear_failed_logins(username: str) -> None:
    _failed_logins.pop(username, None)


def _user_dict(row) -> dict:
    user_id, username, role, is_active, request_limit, token_limit = row
    return {
        "user_id": user_id, "username": username, "role": role, "is_active": is_active,
        "daily_request_limit": request_limit, "daily_token_limit": token_limit,
    }


def create_user(conn, username: str, password: str, role: str = "user") -> dict:
    if not USERNAME_PATTERN.match(username):
        raise AuthError("Username must be 3-32 characters: letters, digits, '_', '.', '-'.")
    if role not in ROLES:
        raise AuthError(f"Role must be one of {ROLES}.")
    password_hash = hash_password(password)
    user_id = str(uuid.uuid4())
    try:
        conn.execute(
            f"INSERT INTO {SCHEMA_NAME}.users (user_id, username, password_hash, role) VALUES (%s, %s, %s, %s)",
            (user_id, username, password_hash, role),
        )
        conn.commit()
    except psycopg.errors.UniqueViolation:
        conn.rollback()
        raise AuthError("That username is already taken.")
    return {"user_id": user_id, "username": username, "role": role}


def authenticate(conn, username: str, password: str, now: float = None) -> str:
    """Verifies credentials and returns a new login token. Failures are
    deliberately generic (no 'unknown user' vs 'wrong password' difference),
    always do a password hash (no timing difference), and are throttled per
    username."""
    check_lockout(username, now)

    row = conn.execute(
        f"SELECT user_id, password_hash, is_active FROM {SCHEMA_NAME}.users WHERE username = %s",
        (username,),
    ).fetchone()

    stored = row[1] if row else _dummy_password_hash()
    valid = verify_password(password, stored)

    if not (row and valid and row[2]):
        record_failed_login(username, now)
        raise AuthError("Invalid username or password.")

    clear_failed_logins(username)
    token = secrets.token_urlsafe(32)
    conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.auth_sessions (token_hash, user_id, expires_at)
            VALUES (%s, %s, now() + make_interval(hours => %s))""",
        (hash_token(token), row[0], token_ttl_hours()),
    )
    conn.commit()
    return token


def get_user_by_token(conn, token: str):
    """The active user a valid, unexpired, unrevoked token belongs to, else None."""
    if not token:
        return None
    row = conn.execute(
        f"""SELECT u.user_id, u.username, u.role, u.is_active, u.daily_request_limit, u.daily_token_limit
            FROM {SCHEMA_NAME}.auth_sessions a
            JOIN {SCHEMA_NAME}.users u ON u.user_id = a.user_id
            WHERE a.token_hash = %s AND a.revoked = FALSE AND a.expires_at > now() AND u.is_active""",
        (hash_token(token),),
    ).fetchone()
    return _user_dict(row) if row else None


def revoke_token(conn, token: str) -> None:
    conn.execute(
        f"UPDATE {SCHEMA_NAME}.auth_sessions SET revoked = TRUE WHERE token_hash = %s",
        (hash_token(token),),
    )
    conn.commit()


def revoke_all_tokens(conn, user_id: str) -> None:
    conn.execute(f"UPDATE {SCHEMA_NAME}.auth_sessions SET revoked = TRUE WHERE user_id = %s", (user_id,))
    conn.commit()


def get_user(conn, username: str):
    row = conn.execute(
        f"""SELECT user_id, username, role, is_active, daily_request_limit, daily_token_limit
            FROM {SCHEMA_NAME}.users WHERE username = %s""",
        (username,),
    ).fetchone()
    return _user_dict(row) if row else None


def list_users(conn) -> list:
    rows = conn.execute(
        f"""SELECT user_id, username, role, is_active, daily_request_limit, daily_token_limit
            FROM {SCHEMA_NAME}.users ORDER BY created_at"""
    ).fetchall()
    return [_user_dict(r) for r in rows]


def update_user(conn, username: str, **fields) -> bool:
    """Admin update of role/active flag/quota limits. A limit of None resets
    it to the role default. Deactivating a user also revokes their tokens."""
    unknown = set(fields) - UPDATABLE_FIELDS
    if unknown:
        raise AuthError(f"Cannot update: {sorted(unknown)}")
    if not fields:
        return False
    if "role" in fields and fields["role"] not in ROLES:
        raise AuthError(f"Role must be one of {ROLES}.")

    assignments = ", ".join(f"{name} = %s" for name in fields)
    cur = conn.execute(
        f"UPDATE {SCHEMA_NAME}.users SET {assignments} WHERE username = %s RETURNING user_id",
        (*fields.values(), username),
    )
    row = cur.fetchone()
    if row is None:
        conn.rollback()
        return False
    if fields.get("is_active") is False:
        conn.execute(f"UPDATE {SCHEMA_NAME}.auth_sessions SET revoked = TRUE WHERE user_id = %s", (row[0],))
    conn.commit()
    return True


def set_password(conn, username: str, new_password: str) -> bool:
    """Sets a new password and signs the user out everywhere."""
    password_hash = hash_password(new_password)
    row = conn.execute(
        f"UPDATE {SCHEMA_NAME}.users SET password_hash = %s WHERE username = %s RETURNING user_id",
        (password_hash, username),
    ).fetchone()
    if row is None:
        conn.rollback()
        return False
    conn.execute(f"UPDATE {SCHEMA_NAME}.auth_sessions SET revoked = TRUE WHERE user_id = %s", (row[0],))
    conn.commit()
    clear_failed_logins(username)
    return True


def ensure_admin_from_env(conn) -> None:
    """Makes the ADMIN_USERNAME / ADMIN_PASSWORD account from .env match .env:
    created if missing; otherwise forced to role admin, active, and the
    password re-synced (only rewritten when it no longer matches, and
    existing logins are revoked then). Does nothing if either is unset."""
    username = os.environ.get("ADMIN_USERNAME", "")
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not username or not password:
        return
    row = conn.execute(
        f"SELECT password_hash FROM {SCHEMA_NAME}.users WHERE username = %s", (username,),
    ).fetchone()
    if row is None:
        create_user(conn, username, password, role="admin")
        return
    update_user(conn, username, role="admin", is_active=True)
    if not verify_password(password, row[0]):
        set_password(conn, username, password)


def is_admin(user: dict) -> bool:
    return bool(user) and user.get("role") == "admin"


def require_admin(user: dict) -> None:
    if not is_admin(user):
        raise PermissionDenied("Admin access required.")
