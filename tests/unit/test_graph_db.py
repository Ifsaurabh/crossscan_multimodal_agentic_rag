import pytest

import graph_db


class FakeDriver:
    def __init__(self, uri, auth, **kwargs):
        self.uri = uri
        self.auth = auth
        self.kwargs = kwargs
        self.closed = 0

    def close(self):
        self.closed += 1


@pytest.fixture(autouse=True)
def fresh_driver_state(monkeypatch):
    monkeypatch.setattr(graph_db, "_driver", None)
    monkeypatch.setattr(graph_db.GraphDatabase, "driver", lambda uri, auth, **kwargs: FakeDriver(uri, auth, **kwargs))
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "testpass")


def test_get_driver_uses_env_credentials():
    driver = graph_db.get_driver()

    assert driver.uri == "bolt://localhost:7687"
    assert driver.auth == ("neo4j", "testpass")


def test_the_driver_has_timeouts_so_a_hung_database_cannot_block_a_request():
    kwargs = graph_db.get_driver().kwargs

    assert kwargs == {
        "connection_timeout": 10, "connection_acquisition_timeout": 15, "liveness_check_timeout": 60,
    }


def test_one_driver_is_shared_for_the_whole_process():
    assert graph_db.get_driver() is graph_db.get_driver()


def test_close_driver_closes_it_and_a_later_call_builds_a_new_one():
    first = graph_db.get_driver()

    graph_db.close_driver()

    assert first.closed == 1
    assert graph_db.get_driver() is not first


def test_closing_when_no_driver_exists_is_harmless():
    graph_db.close_driver()
