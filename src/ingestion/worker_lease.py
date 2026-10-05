"""worker_lease: only ONE worker run handles the queue at a time.

A Cloud Run job can be started many times at once (for example one execution per uploaded file). Each execution first asks
for the lease, a single row in Postgres:
  - the first one gets it and handles the queue until it is empty;
  - the others see the lease is taken and exit at once, before loading any model, so they cost almost nothing.

The holder renews the lease every RENEW_SECONDS in a background thread. If the holder dies (killed, timed out), the lease
runs out after TTL_SECONDS and the next execution takes over; the message that was being handled is delivered again by Pub/Sub.
If the holder finds it has lost the lease (a long database outage), `lease.lost` is set and the worker stops taking new messages.

Why exiting at once cannot leave a file unprocessed: the holder only gives the lease back after two empty pulls (about a minute),
and an execution started by a later upload needs several seconds just to start, so by the time it asks, the lease is free. A
message that still slips through stays in the queue (it is kept for days) and the next upload or manual run handles it.
"""
import contextlib
import os
import socket
import threading
import uuid

from shared.db import SCHEMA_NAME

LEASE_NAME = "ingest"
TTL_SECONDS = 300
RENEW_SECONDS = 60


def setup_sql(schema: str) -> str:
    """The SQL that creates the lease table in `schema`. Only creates."""
    return f"""
CREATE TABLE IF NOT EXISTS {schema}.worker_lease (
    name TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);
"""


def holder_id() -> str:
    """Who is asking: the Cloud Run execution when there is one, else the machine and process."""
    where = os.environ.get("CLOUD_RUN_EXECUTION") or f"{socket.gethostname()}-{os.getpid()}"
    return f"{where}-{uuid.uuid4().hex[:6]}"


def try_acquire(conn, holder: str, name: str = LEASE_NAME, ttl: float = TTL_SECONDS) -> bool:
    """Takes the lease if it is free, expired, or already ours. True when we hold it afterwards."""
    row = conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.worker_lease (name, holder, acquired_at, expires_at)
            VALUES (%s, %s, now(), now() + make_interval(secs => %s))
            ON CONFLICT (name) DO UPDATE
                SET holder = EXCLUDED.holder, acquired_at = now(), expires_at = EXCLUDED.expires_at
                WHERE {SCHEMA_NAME}.worker_lease.expires_at < now() OR {SCHEMA_NAME}.worker_lease.holder = EXCLUDED.holder
            RETURNING holder""",
        (name, holder, ttl),
    ).fetchone()
    return row is not None and row[0] == holder


def renew(conn, holder: str, name: str = LEASE_NAME, ttl: float = TTL_SECONDS) -> bool:
    """Pushes the expiry forward. False when the lease is no longer ours."""
    row = conn.execute(
        f"""UPDATE {SCHEMA_NAME}.worker_lease SET expires_at = now() + make_interval(secs => %s)
            WHERE name = %s AND holder = %s RETURNING 1""",
        (ttl, name, holder),
    ).fetchone()
    return row is not None


def release(conn, holder: str, name: str = LEASE_NAME) -> None:
    """Gives the lease back (only if it is ours)."""
    conn.execute(f"DELETE FROM {SCHEMA_NAME}.worker_lease WHERE name = %s AND holder = %s", (name, holder))


class Lease:
    def __init__(self, holder: str):
        self.holder = holder
        self.acquired = False
        self.lost = threading.Event()  # set when a renewal finds the lease is no longer ours


@contextlib.contextmanager
def hold(connect=None, holder: str = None, name: str = LEASE_NAME, ttl: float = TTL_SECONDS, renew_every: float = RENEW_SECONDS):
    """`with hold() as lease:` then check `lease.acquired`. While the block runs, a thread renews the lease; it is given back
    when the block ends, however it ends. `connect` gives a connection context (default: the shared pool)."""
    if connect is None:
        from shared.db import connection as connect

    lease = Lease(holder or holder_id())
    with connect() as conn:
        lease.acquired = try_acquire(conn, lease.holder, name, ttl)
    if not lease.acquired:
        yield lease
        return

    stop = threading.Event()

    def heartbeat():
        while not stop.wait(renew_every):
            try:
                with connect() as conn:
                    if not renew(conn, lease.holder, name, ttl):
                        lease.lost.set()
                        return
            except Exception as e:  # a hiccup: try again at the next beat; the lease still has time left
                print(f"   (lease renewal failed, will retry: {type(e).__name__}: {e})")

    thread = threading.Thread(target=heartbeat, name="worker-lease", daemon=True)
    thread.start()
    try:
        yield lease
    finally:
        stop.set()
        thread.join(timeout=5)
        try:
            with connect() as conn:
                release(conn, lease.holder, name)
        except Exception as e:
            print(f"   (could not give the lease back, it will run out by itself: {type(e).__name__}: {e})")
