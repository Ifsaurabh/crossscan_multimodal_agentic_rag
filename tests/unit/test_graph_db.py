import graph_db


def test_get_driver_uses_env_credentials(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "testpass")

    seen = {}

    def fake_driver(uri, auth):
        seen["uri"] = uri
        seen["auth"] = auth
        return "fake-driver-object"

    monkeypatch.setattr(graph_db.GraphDatabase, "driver", fake_driver)

    result = graph_db.get_driver()

    assert result == "fake-driver-object"
    assert seen["uri"] == "bolt://localhost:7687"
    assert seen["auth"] == ("neo4j", "testpass")
