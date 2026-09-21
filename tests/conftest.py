import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
SRC_DIR = TESTS_DIR.parent / "src"
for path in (SRC_DIR, TESTS_DIR):  # tests/ too, so subfolders can import `factories` / `fake_db`
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Unit tests must never reach a real Langfuse server (prompt fetch/tracing) -
# set before any src module loads .env.
os.environ["LANGFUSE_DISABLED"] = "1"
# ...and online evaluation must not randomly judge answers (it would spawn a
# thread and try a real model call). Tests of it pass their own rate.
os.environ["ONLINE_EVAL_SAMPLE_RATE"] = "0"


# ---------- test kinds (folders = what a test NEEDS; CI decides WHEN it runs) ----------
#   tests/unit/         nothing external, everything faked         always runs
#   tests/integration/  the REAL Postgres, in a throwaway schema   opt-in: --run-integration
#   tests/live/         REAL Gemini / Neo4j / full pipeline        opt-in: --run-live (spends quota)
# Opt-in tests are skipped unless asked, by flag or by environment variable
# (RUN_INTEGRATION=1 / RUN_LIVE=1).

OPT_IN_KINDS = {
    "integration": ("--run-integration", "RUN_INTEGRATION", "uses the real Postgres"),
    "live": ("--run-live", "RUN_LIVE", "uses real Gemini/Neo4j and spends quota"),
}


def pytest_addoption(parser):
    for kind, (flag, _, why) in OPT_IN_KINDS.items():
        parser.addoption(flag, action="store_true", default=False, help=f"also run {kind} tests ({why})")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: uses the real Postgres in a throwaway schema; opt-in via --run-integration",
    )
    config.addinivalue_line(
        "markers", "live: uses real Gemini/Neo4j and spends quota; opt-in via --run-live",
    )


def pytest_collection_modifyitems(config, items):
    """Marks tests by folder (so a file needs no marker of its own), then skips
    the opt-in kinds that were not asked for."""
    for item in items:
        parts = item.path.parts
        for kind in OPT_IN_KINDS:
            if kind in parts[-3:-1] and kind not in item.keywords:
                item.add_marker(getattr(pytest.mark, kind))

    for kind, (flag, env_name, why) in OPT_IN_KINDS.items():
        if config.getoption(flag) or os.environ.get(env_name) == "1":
            continue
        skip = pytest.mark.skip(reason=f"{kind} test: run with {flag} ({why})")
        for item in items:
            if kind in item.keywords:
                item.add_marker(skip)


# ---------- Faker ----------

@pytest.fixture(scope="session")
def faker_seed():
    """Fixed seed for Faker's pytest fixture: realistic fake data, identical on
    every run, so any failure is reproducible."""
    return 20260920


# ---------- guards: unit tests must not reach real services ----------

def _blocked(service: str):
    def blocked(*args, **kwargs):
        raise RuntimeError(
            f"A unit test tried to reach the real {service}. Fake it (monkeypatch / a stand-in), "
            f"or, if it really needs the live service, mark it @pytest.mark.integration."
        )
    return blocked


@pytest.fixture(autouse=True)
def _block_real_services(request, monkeypatch):
    """Postgres, Neo4j and Gemini are off-limits to unit tests. A test that
    forgets to fake one fails loudly instead of quietly hitting a real service
    (and, for Gemini, spending quota). Tests that DO fake them just override
    these patches. Integration and live tests are exempt."""
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("live"):
        yield
        return

    import neo4j
    import psycopg
    from google import genai

    monkeypatch.setattr(psycopg, "connect", _blocked("Postgres"))
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", _blocked("Neo4j"))
    monkeypatch.setattr(genai, "Client", _blocked("Gemini API"))
    yield


@pytest.fixture(autouse=True)
def _no_real_llm_fallbacks(monkeypatch):
    """The Anthropic/OpenAI fallbacks must never make real (billed) calls from
    a unit test, even when their keys are present in .env. Tests that exercise
    llm_connection replace these adapters themselves."""
    import llm_connection

    def blocked(*args, **kwargs):
        raise llm_connection.ProviderUnavailable("real fallback providers are blocked in unit tests")

    monkeypatch.setitem(llm_connection._ADAPTERS, "anthropic", blocked)
    monkeypatch.setitem(llm_connection._ADAPTERS, "openai", blocked)
    llm_connection.reset_cooldowns()
    yield
    llm_connection.reset_cooldowns()
