def require_database_url(database_url: str | None) -> str:
    if not database_url:
        raise RuntimeError("缺少 DATABASE_URL 环境变量")
    return database_url


def sqlalchemy_database_url(database_url: str | None) -> str:
    """Use the installed Psycopg 3 driver for SQLAlchemy PostgreSQL URLs."""

    url = require_database_url(database_url)
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def psycopg_connection_url(database_url: str | None) -> str:
    """Remove SQLAlchemy's driver suffix for direct Psycopg connections."""

    url = require_database_url(database_url)
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://"):
        if url.startswith(prefix):
            return "postgresql://" + url.removeprefix(prefix)
    if url.startswith("postgres://"):
        return "postgresql://" + url.removeprefix("postgres://")
    return url
