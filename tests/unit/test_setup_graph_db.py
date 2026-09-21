import setup_graph_db as sgd


class FakeSession:
    def __init__(self):
        self.queries = []

    def run(self, query, **kwargs):
        self.queries.append(query)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeDriver:
    def __init__(self):
        self.session_obj = FakeSession()
        self.closed = False

    def session(self):
        return self.session_obj

    def close(self):
        self.closed = True


def test_setup_graph_db_applies_all_constraints(monkeypatch):
    fake_driver = FakeDriver()
    monkeypatch.setattr(sgd, "get_driver", lambda: fake_driver)

    sgd.setup_graph_db()

    assert len(fake_driver.session_obj.queries) == 6
    assert fake_driver.closed
    labels = ["Paper", "Method", "Dataset", "Metric", "Image", "Section"]
    for label, query in zip(labels, fake_driver.session_obj.queries):
        assert f"FOR ({label[0].lower()}:{label})" in query
