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
    register_vector(conn)
    conn.execute(f"SET search_path TO {SCHEMA_NAME}, public")
    return conn
