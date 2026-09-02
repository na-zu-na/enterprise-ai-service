from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from core.config import Settings


@asynccontextmanager
async def checkpoint_saver() -> AsyncIterator[AsyncPostgresSaver]:
    """Create the application-wide PostgreSQL checkpoint saver."""

    pool = AsyncConnectionPool(
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
    await pool.open(wait=True)

    try:
        saver = AsyncPostgresSaver(pool)
        # Idempotent: creates or migrates the LangGraph checkpoint tables.
        await saver.setup()
        yield saver
    finally:
        await pool.close()
