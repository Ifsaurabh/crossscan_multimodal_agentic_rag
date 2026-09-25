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

    def session(self):
        return self.session_obj


def test_setup_graph_db_applies_all_constraints(monkeypatch):
    fake_driver = FakeDriver()
    closed = []
    monkeypatch.setattr(sgd, "get_driver", lambda: fake_driver)
    monkeypatch.setattr(sgd, "close_driver", lambda: closed.append(True))

    sgd.setup_graph_db()

    assert len(fake_driver.session_obj.queries) == 8
    assert closed == [True]  # the one-shot script releases the shared driver when it finishes
    labels = ["Paper", "Method", "Dataset", "Metric", "Image", "Section", "Table", "Baseline"]
    for label, query in zip(labels, fake_driver.session_obj.queries):
        assert f"FOR ({label[0].lower()}:{label})" in query


def test_every_constraint_is_idempotent():
    assert all("IF NOT EXISTS" in constraint for constraint in sgd.CONSTRAINTS)


def test_table_and_baseline_nodes_are_unique_by_the_keys_the_loader_merges_on():
    joined = "\n".join(sgd.CONSTRAINTS)

    assert "(t:Table) REQUIRE t.table_id IS UNIQUE" in joined
    assert "(b:Baseline) REQUIRE b.name IS UNIQUE" in joined
