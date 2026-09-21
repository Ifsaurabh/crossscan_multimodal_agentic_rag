import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

SCHEMA_NAME = os.environ.get("SCHEMA_NAME", "rag_new")


def get_connection():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    # search_path must be set BEFORE register_vector(): it looks up the "vector" type
    # unqualified, which only resolves if "public" (where the extension lives) is on the
    # search path. Some managed Postgres roles (seen on Neon) default to an EMPTY search_path,
    # unlike a typical local install ("$user", public) - the old order worked locally only by
    # accident and raised psycopg.ProgrammingError("vector type not found") on Neon.
    conn.execute(f"SET search_path TO {SCHEMA_NAME}, public")
    register_vector(conn)
    return conn
