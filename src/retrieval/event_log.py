"""Time-stamped events per (bucket, key): what the per-minute rate limit, the daily sign-up cap and the login lockout count.

Two stores with the same methods:
- PgEventLog keeps the events in Postgres, so every Cloud Run instance sees the same counts and a restart forgets nothing.
- MemoryEventLog keeps them in the process (tests, and a fallback for scripts).

Times are epoch seconds from the caller. A check-and-add is one step (an advisory lock on the key in Postgres), so two
instances cannot both take the last slot. `conn` is the caller's connection when it has one; without it a connection is
borrowed from the pool for the call."""
import contextlib
import threading
from collections import defaultdict

from shared.db import SCHEMA_NAME, connection


class MemoryEventLog:
    def __init__(self):
        self._events = defaultdict(list)
        self._lock = threading.Lock()

    def _recent(self, bucket, key, since):
        recent = [t for t in self._events[(bucket, key)] if t > since]
        self._events[(bucket, key)] = recent
        return recent

    def try_add(self, bucket, key, limit, since, now, conn=None):
        """(allowed, oldest event in the window). Adds the event only when it is allowed."""
        with self._lock:
            recent = self._recent(bucket, key, since)
            if len(recent) >= limit:
                return False, recent[0]
            recent.append(now)
            return True, None

    def add(self, bucket, key, now, conn=None):
        with self._lock:
            self._events[(bucket, key)].append(now)

    def count(self, bucket, key, since, conn=None):
        """(events in the window, the oldest of them or None)."""
        with self._lock:
            recent = self._recent(bucket, key, since)
            return len(recent), (recent[0] if recent else None)

    def remove_latest(self, bucket, key, conn=None):
        with self._lock:
            if self._events[(bucket, key)]:
                self._events[(bucket, key)].pop()

    def clear(self, bucket, key, conn=None):
        with self._lock:
            self._events.pop((bucket, key), None)


class PgEventLog:
    @contextlib.contextmanager
    def _connection(self, conn):
        if conn is not None:
            yield conn
            return
        with connection() as borrowed:
            yield borrowed

    @staticmethod
    def _lock(conn, bucket, key):
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{bucket}:{key}",))

    @staticmethod
    def _stats(conn, bucket, key, since):
        row = conn.execute(
            f"SELECT COUNT(*), MIN(at_epoch) FROM {SCHEMA_NAME}.limit_events WHERE bucket = %s AND key = %s AND at_epoch > %s",
            (bucket, key, since)).fetchone()
        return int(row[0]), row[1]

    def try_add(self, bucket, key, limit, since, now, conn=None):
        with self._connection(conn) as c:
            self._lock(c, bucket, key)
            c.execute(f"DELETE FROM {SCHEMA_NAME}.limit_events WHERE bucket = %s AND key = %s AND at_epoch <= %s", (bucket, key, since))
            n, oldest = self._stats(c, bucket, key, since)
            if n >= limit:
                c.commit()
                return False, oldest
            c.execute(f"INSERT INTO {SCHEMA_NAME}.limit_events (bucket, key, at_epoch) VALUES (%s, %s, %s)", (bucket, key, now))
            c.commit()
            return True, None

    def add(self, bucket, key, now, conn=None):
        with self._connection(conn) as c:
            c.execute(f"INSERT INTO {SCHEMA_NAME}.limit_events (bucket, key, at_epoch) VALUES (%s, %s, %s)", (bucket, key, now))
            c.commit()

    def count(self, bucket, key, since, conn=None):
        with self._connection(conn) as c:
            result = self._stats(c, bucket, key, since)
            c.commit()
            return result

    def remove_latest(self, bucket, key, conn=None):
        with self._connection(conn) as c:
            self._lock(c, bucket, key)
            c.execute(
                f"""DELETE FROM {SCHEMA_NAME}.limit_events WHERE event_id = (
                        SELECT event_id FROM {SCHEMA_NAME}.limit_events WHERE bucket = %s AND key = %s
                        ORDER BY at_epoch DESC, event_id DESC LIMIT 1)""", (bucket, key))
            c.commit()

    def clear(self, bucket, key, conn=None):
        with self._connection(conn) as c:
            c.execute(f"DELETE FROM {SCHEMA_NAME}.limit_events WHERE bucket = %s AND key = %s", (bucket, key))
            c.commit()
