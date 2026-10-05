import os

from shared.db import DEFAULT_POOL_MAX
from shared.model_config import GEMINI_MODEL  # noqa: F401  (defined once, in shared/, and re-exported here)
RERANKER_MODEL = "BAAI/bge-reranker-base"

MAX_RETRY_ATTEMPTS = 3
EXPANSION_VARIANTS = 3
# Decision (user, 2026-10-05): a question is split into at most this many sub-questions (each one is a full search
# plus rerank, so an uncapped split could spend many model calls on one message).
MAX_SUB_QUERIES = 3

VECTOR_TOP_K = 5
# A question that asks WHICH papers do something needs passages from many papers,
# so it is searched more widely.
LISTING_TOP_K = 15

# Tables in the answer prompt
TABLE_TOP_K = 3                  # tables returned by the table search when the question asks for a table
MIN_TABLE_SIMILARITY = 0.45      # a table below this is not used (starting value: tune it on the golden set)
MAX_TABLES = 4                   # tables per sub-question, from the parents' tables and the table search together
MAX_TABLE_TOKENS = 1500          # one table: a longer one is cut after its header and first rows, with a note
MAX_TABLES_TOKENS = 4000         # all the tables of a sub-question
CHARS_PER_TOKEN = 4              # the usual estimate for English text, so no tokenizer is needed in the app

# Parallel retrieval: at most this many database searches run at once in one process. It is the size of
# the connection pool (DB_POOL_MAX, default 8), so searches never starve the pool; the other queries of a turn (cache,
# images, tables) borrow connections too and wait their turn.
MAX_PARALLEL_SEARCHES = int(os.environ.get("DB_POOL_MAX") or DEFAULT_POOL_MAX)
