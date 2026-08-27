from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from core.config import Settings


@contextmanager
def checkpoint_saver() -> Iterator[PostgresSaver]:
    """Create the application-wide PostgreSQL checkpoint saver."""

    pool = ConnectionPool(
        conninfo=Settings.DATABASE_URL,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    pool.open(wait=True)

    try:
        saver = PostgresSaver(pool)
        # Idempotent: creates or migrates the LangGraph checkpoint tables.
        saver.setup()
        yield saver
    finally:
        pool.close()
