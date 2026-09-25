import os
import threading

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

_driver = None
_driver_lock = threading.Lock()


def get_driver():
    """The one shared Neo4j driver for the whole process. A driver is
    thread-safe and keeps its own pool of connections, so it is created once
    and never closed per call (building a new one for every lookup meant a new
    secure connection to Aura each time). Timeouts stop a hung Aura from
    blocking a request; liveness_check_timeout re-tests a connection that sat
    idle before using it."""
    global _driver
    with _driver_lock:
        if _driver is None:
            _driver = GraphDatabase.driver(
                os.environ["NEO4J_URI"],
                auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
                connection_timeout=10,
                connection_acquisition_timeout=15,
                liveness_check_timeout=60,
            )
        return _driver


def close_driver() -> None:
    """For one-shot scripts to call when they finish."""
    global _driver
    with _driver_lock:
        if _driver is not None:
            _driver.close()
            _driver = None
