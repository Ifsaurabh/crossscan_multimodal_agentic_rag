import os
import threading

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

load_dotenv()

SCHEMA_NAME = os.environ.get("SCHEMA_NAME", "rag_new")

# Pool sizing, all overridable from .env. A chat request holds one connection
# for its whole turn and its graph nodes borrow more meanwhile, so each running
# request can hold two: 3 concurrent requests (MAX_CONCURRENT_REQUESTS) need 6,
# plus the background online-eval thread and sidebar page loads.
DEFAULT_POOL_MIN = 2
DEFAULT_POOL_MAX = 8
DEFAULT_POOL_TIMEOUT_SECONDS = 15   # how long a caller waits for a free connection
CONNECT_TIMEOUT_SECONDS = 10        # how long opening ONE new connection may take

_pool = None
_pool_lock = threading.Lock()


def _set_up_connection(conn):
    """Runs once for each NEW physical connection (not on every borrow).
    search_path must be set BEFORE register_vector(): it looks up the "vector"
    type unqualified, which only resolves if "public" (where the extension
    lives) is on the search path. Some managed Postgres roles (seen on Neon)
    default to an EMPTY search_path, unlike a typical local install
    ("$user", public). The pool needs the connection back in an idle state,
    hence the commit."""
    conn.execute(f"SET search_path TO {SCHEMA_NAME}, public")
    register_vector(conn)
    conn.commit()


def get_pool() -> ConnectionPool:
    """The shared, thread-safe connection pool, created on first use. Opening
    a Neon connection costs seconds (TLS, login, setup), so connections are
    opened once and reused. `check` tests a connection before it is lent out,
    which replaces one Neon silently closed after its idle auto-suspend (the
    'SSL connection has been closed unexpectedly' error)."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                conninfo=os.environ["DATABASE_URL"],
                kwargs={"connect_timeout": CONNECT_TIMEOUT_SECONDS},
                min_size=int(os.environ.get("DB_POOL_MIN") or DEFAULT_POOL_MIN),
                max_size=int(os.environ.get("DB_POOL_MAX") or DEFAULT_POOL_MAX),
                timeout=float(os.environ.get("DB_POOL_TIMEOUT") or DEFAULT_POOL_TIMEOUT_SECONDS),
                configure=_set_up_connection,
                check=ConnectionPool.check_connection,
                open=False,
            )
            _pool.open()  # returns at once; the minimum connections fill in the background
        return _pool


def connection():
    """Borrow a pooled connection for a `with` block; it goes back to the pool
    afterwards (committed on success, rolled back on an error). Never call
    .close() on it. This is what the running application uses."""
    return get_pool().connection()


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def get_connection():
    """A single, direct (un-pooled) connection that the CALLER closes. For
    one-shot scripts (ingestion, setup, admin CLI) where pooling gains nothing.
    Also the only place the pgvector extension is created, so a fresh database
    gets it the first time any script connects."""
    conn = psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=CONNECT_TIMEOUT_SECONDS)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    _set_up_connection(conn)
    return conn
