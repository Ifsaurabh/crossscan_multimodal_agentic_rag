"""Property-style tests: many realistic random inputs (Faker, fixed seed) instead
of a handful of hand-picked ones. They cover the pure, deterministic logic."""
import auth
import factories
import ingestion_guardrails
import quotas
import query_guardrail

RUNS = 50


# ---------- email redaction ----------

def test_query_guardrail_redacts_every_random_email(faker):
    for _ in range(RUNS):
        address = factories.email(faker)
        cleaned, count = query_guardrail.redact_pii(f"{faker.sentence()} {address} {faker.sentence()}")

        assert address not in cleaned and count == 1


def test_ingestion_guardrail_redacts_every_random_email(faker):
    for _ in range(RUNS):
        address = factories.email(faker)
        cleaned, count = ingestion_guardrails.redact_pii(f"Contact {address} for the data.")

        assert address not in cleaned and count == 1


def test_text_without_an_email_is_returned_unchanged(faker):
    for _ in range(RUNS):
        text = faker.sentence(nb_words=12).replace("@", "")

        assert query_guardrail.redact_pii(text) == (text, 0)


# ---------- usernames and passwords ----------

def test_generated_usernames_always_satisfy_the_username_rule(faker):
    for _ in range(RUNS):
        assert auth.USERNAME_PATTERN.match(factories.username(faker))


def test_usernames_with_spaces_or_symbols_are_rejected(faker):
    for _ in range(RUNS):
        bad = f"{faker.first_name()} {faker.last_name()}!"

        assert not auth.USERNAME_PATTERN.match(bad)


def test_random_passwords_verify_only_against_themselves(faker):
    for _ in range(5):  # scrypt is deliberately slow
        plain, other = factories.password(faker), factories.password(faker)
        stored = auth.hash_password(plain)

        assert auth.verify_password(plain, stored) is True
        assert auth.verify_password(other, stored) is False
        assert plain not in stored


def test_same_password_hashes_differently_each_time(faker):
    plain = factories.password(faker)

    assert auth.hash_password(plain) != auth.hash_password(plain)  # random salt


# ---------- quota decision ----------

def test_quota_evaluate_matches_the_rules_for_random_usage(faker):
    for _ in range(RUNS * 4):
        usage = {
            "requests": faker.random_int(0, 20),
            "prompt_tokens": faker.random_int(0, 5000),
            "output_tokens": faker.random_int(0, 5000),
        }
        request_limit = faker.random_element([None, faker.random_int(1, 20)])
        token_limit = faker.random_element([None, faker.random_int(1, 10000)])
        cap = faker.random_element([0, faker.random_int(1, 60)])
        global_requests = faker.random_int(0, 80)

        reason = quotas.evaluate(usage, (request_limit, token_limit), global_requests, cap)

        if request_limit is not None and usage["requests"] >= request_limit:
            expected = "daily_request_limit"
        elif token_limit is not None and usage["prompt_tokens"] + usage["output_tokens"] >= token_limit:
            expected = "daily_token_limit"
        elif cap and global_requests >= cap:
            expected = "global_daily_cap"
        else:
            expected = None
        assert reason == expected


def test_no_limits_and_no_cap_never_refuses(faker):
    usage = {"requests": faker.random_int(0, 10**6), "prompt_tokens": 10**6, "output_tokens": 10**6}

    assert quotas.evaluate(usage, (None, None), faker.random_int(0, 10**6), 0) is None
