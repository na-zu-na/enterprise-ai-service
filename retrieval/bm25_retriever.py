from clients.elasticsearch_client import (
    ReadOnlyElasticsearchClient,
    get_elasticsearch_client,
)
from core.config import Settings
from retrieval.models import RetrievalCandidate
from schemas.embedding import VectorRetrievalRequest


class Bm25Retriever:
    def __init__(
            self,
            index_name: str | None = None,
            client: ReadOnlyElasticsearchClient | None = None,
    ) -> None:
        self._client = client or get_elasticsearch_client()
        self._index_name = (
            index_name or Settings.ELASTICSEARCH_INDEX
        )

    def retrieve(
            self,
            request: VectorRetrievalRequest,
    ) -> list[RetrievalCandidate]:
        if not request.knowledge_base_ids:
            return []

        query = {
            "bool": {
                "filter": [
                    {
                        "terms": {
                            "knowledge_base_id": (
                                request.knowledge_base_ids
                            )
                        }
                    }
                ],
                "must": [
                    {
                        "multi_match": {
                            "query": request.query,
                            "fields": [
                                "section_title^3",
                                "document_name^2",
                                "content",
                            ],
                            "type": "best_fields",
                        }
                    }
                ],
            }
        }

        response = self._client.search(
            index=self._index_name,
            query=query,
            size=request.candidate_k,
        )

        candidates: list[RetrievalCandidate] = []

        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            candidates.append(
                RetrievalCandidate(
                    id=int(source["id"]),
                    document_id=int(source["document_id"]),
                    knowledge_base_id=int(
                        source["knowledge_base_id"]
                    ),
                    chunk_index=int(source["chunk_index"]),
                    content=source["content"],
                    document_name=source.get(
                        "document_name",
                        "",
                    ),
                    section_title=source.get(
                        "section_title",
                        "",
                    ) or "",
                    bm25_score=float(hit["_score"]),
                )
            )

        return candidates
