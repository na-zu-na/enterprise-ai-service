import os

class Settings:
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "3021bc3c-348f-481f-ae00-25ecec9dc467")
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

    RERANKER_USE_FP16: bool = (
            os.getenv("RERANKER_USE_FP16", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
    )
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
