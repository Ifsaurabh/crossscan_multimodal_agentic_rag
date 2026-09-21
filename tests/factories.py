"""Realistic fake data for tests, built on Faker.

Faker supplies the VALUES (people, emails, sentences); the tests still fake
the BEHAVIOUR of services (Gemini, Postgres, Neo4j) with stand-ins. The seed is
fixed in conftest.py (`faker_seed`), so every run produces the same data and a
failure is always reproducible.

Helpers take the `faker` pytest fixture as their first argument.
"""
import re

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def username(faker) -> str:
    """A valid app username: 3-32 characters from letters, digits, '_', '.', '-'."""
    base = _UNSAFE.sub("", faker.user_name()) or "user"
    return f"{base[:20]}{faker.random_int(100, 999)}"


def password(faker, length: int = 14) -> str:
    return faker.password(length=length)


def email(faker) -> str:
    """An ASCII email address the redaction patterns are meant to match."""
    return f"{username(faker)}@{faker.domain_name()}"


def message(faker) -> str:
    return faker.sentence(nb_words=8)


def user_dict(faker, role: str = "user", **overrides) -> dict:
    """The user dict the app passes around (as returned by auth.get_user)."""
    user = {
        "user_id": faker.uuid4(),
        "username": username(faker),
        "role": role,
        "is_active": True,
        "daily_request_limit": None,
        "daily_token_limit": None,
    }
    user.update(overrides)
    return user
