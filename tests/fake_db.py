class FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class FakeConn:
    """Stand-in for a psycopg connection. Records every (sql, params) and
    answers a statement with the rows of the FIRST (needle, rows) response
    whose needle appears in the SQL text; anything else returns no rows."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.executed = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        for needle, rows in self.responses:
            if needle in sql:
                return FakeCursor(rows)
        return FakeCursor([])

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True

    def find(self, needle):
        """All (sql, params) executed whose SQL contains needle."""
        return [(sql, params) for sql, params in self.executed if needle in sql]
