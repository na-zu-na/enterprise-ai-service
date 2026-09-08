import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    return (
        os.getenv(name, str(default))
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )

class Settings:
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-v4-flash-ga-260731")

    ELASTICSEARCH_URL: str = os.getenv(
        "ELASTICSEARCH_URL",
        "http://localhost:9200",
    )
    ELASTICSEARCH_INDEX: str = os.getenv(
        "ELASTICSEARCH_INDEX",
        "document_chunk",
    )
    ELASTICSEARCH_API_KEY: str | None = os.getenv(
        "ELASTICSEARCH_API_KEY"
    )
    ELASTICSEARCH_CA_CERTS: str | None = os.getenv(
        "ELASTICSEARCH_CA_CERTS"
    )

    RERANKER_USE_FP16: bool = _env_bool("RERANKER_USE_FP16")
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL",
        "BAAI/bge-reranker-v2-m3",
    )
    RERANKER_QUERY_MAX_LENGTH: int = int(
        os.getenv("RERANKER_QUERY_MAX_LENGTH", "256")
    )

    RERANKER_PASSAGE_MAX_LENGTH: int = int(
        os.getenv("RERANKER_PASSAGE_MAX_LENGTH", "512")
    )
    RERANKER_MIN_SCORE: float = float(
        os.getenv("RERANKER_MIN_SCORE", "0.5")
    )

    SPRING_REQUEST_URL: str = os.getenv(
        "SPRING_REQUEST_URL","http://localhost:8080"
    )

    SPRING_KNOWLEDGE_BASE_IDS_PATH: str = os.getenv(
        "SPRING_KNOWLEDGE_BASE_IDS_PATH",
        "/api/knowledge-bases/accessible-ids",
    )

    SPRING_USER_NAME: str | None = os.getenv("SPRING_USER_NAME")

    SPRING_USER_PASSWORD: str | None = os.getenv("SPRING_USER_PASSWORD")

    DATABASE_URL: str | None = os.getenv("DATABASE_URL")

    GOOGLE_CALENDAR_MCP_PYTHON: str | None = os.getenv(
        "GOOGLE_CALENDAR_MCP_PYTHON"
    )
    GOOGLE_CALENDAR_MCP_SERVER: str | None = os.getenv(
        "GOOGLE_CALENDAR_MCP_SERVER"
    )

    TEST_ACCESS_TOKEN: str | None = os.getenv("TEST_ACCESS_TOKEN")
