"""The limits must fit each other. Each test is one relation between two limits, so changing one limit breaks the test of the
limit it no longer fits."""
import math

from retrieval import quotas
from retrieval import retrieval_config as cfg
from shared import db

# Facts of the live setup (read from Cloud Run and Neon on 2026-10-05); change them here when the setup changes.
CLOUD_RUN_MAX_INSTANCES = 5   # set by the deploy workflow (--max-instances 5)
CLOUD_RUN_REQUESTS_PER_INSTANCE = 80
NEON_MAX_CONNECTIONS = 901
SUB_QUESTIONS_PER_QUESTION = 3   # what the pool is sized for; the planner has no hard cap yet
ONLINE_EVAL_THREADS = 1
BUFFER = 1.05


def test_the_users_served_per_day_fit_inside_the_global_request_cap():
    per_user = quotas._ROLE_DEFAULTS["user"][0]
    assert quotas.DEFAULT_DAILY_USER_CAP * per_user <= quotas.DEFAULT_GLOBAL_DAILY_CAP   # 10 x 6 = 60 <= 100


def test_the_per_minute_limit_is_below_the_daily_limit():
    assert quotas.DEFAULT_RATE_LIMIT_PER_MINUTE <= quotas._ROLE_DEFAULTS["user"][0]


def test_the_pool_covers_the_worst_case_of_the_concurrency_limit_plus_five_percent():
    searches_per_question = SUB_QUESTIONS_PER_QUESTION * cfg.EXPANSION_VARIANTS
    needed = quotas.DEFAULT_MAX_CONCURRENT * (searches_per_question + 1) + ONLINE_EVAL_THREADS   # 3 x (9 + 1) + 1 = 31
    assert db.DEFAULT_POOL_MAX >= math.ceil(needed * BUFFER) == 33


def test_the_parallel_search_limit_is_the_pool_size_so_searches_never_starve_the_pool():
    assert cfg.MAX_PARALLEL_SEARCHES == db.DEFAULT_POOL_MAX


def test_all_instances_together_stay_inside_the_databases_connection_limit():
    assert CLOUD_RUN_MAX_INSTANCES * db.DEFAULT_POOL_MAX <= NEON_MAX_CONNECTIONS


def test_the_in_app_concurrency_limit_is_stricter_than_what_cloud_run_sends_an_instance():
    assert quotas.DEFAULT_MAX_CONCURRENT <= CLOUD_RUN_REQUESTS_PER_INSTANCE


def test_one_question_cannot_use_more_searches_at_once_than_the_pool_holds():
    assert SUB_QUESTIONS_PER_QUESTION * cfg.EXPANSION_VARIANTS <= db.DEFAULT_POOL_MAX


def test_the_database_wait_is_shorter_than_a_request_can_take():
    assert db.DEFAULT_POOL_TIMEOUT_SECONDS < 300   # Cloud Run's request timeout


def test_the_deploy_workflow_sets_the_instance_maximum_the_alignment_tests_assume():
    from pathlib import Path

    workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert f"--max-instances {CLOUD_RUN_MAX_INSTANCES}" in workflow
