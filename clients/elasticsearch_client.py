from functools import lru_cache
from typing import Any

from elasticsearch import Elasticsearch

from core.config import Settings


class ReadOnlyElasticsearchClient:
    def __init__(self) -> None:
        options: dict[str, Any] = {
            "hosts": [Settings.ELASTICSEARCH_URL],
            "request_timeout": 5,
            "retry_on_timeout": True,
            "max_retries": 2,
            "http_compress": True,
        }

        if Settings.ELASTICSEARCH_API_KEY:
            options["api_key"] = Settings.ELASTICSEARCH_API_KEY

        if Settings.ELASTICSEARCH_CA_CERTS:
            options["ca_certs"] = Settings.ELASTICSEARCH_CA_CERTS

        self._client = Elasticsearch(**options)

    def search(
            self,
            *,
            index: str,
            query: dict[str, Any],
            size: int,
    ) -> dict[str, Any]:
        response = self._client.search(
            index=index,
            query=query,
            size=size,
        )

        return response.body

    def close(self) -> None:
        self._client.close()


@lru_cache(maxsize=1)
def get_elasticsearch_client() -> ReadOnlyElasticsearchClient:
    return ReadOnlyElasticsearchClient()
