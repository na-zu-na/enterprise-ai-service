from sqlalchemy.orm import Session

from schemas.embedding import VectorRetrievalRequest, HybridRetrievalResponse
from services.hybrid_retrieval_service import HybridRetrievalService
from services.reranking_service import RerankingService


class RetrievalPipelineService:
    def __init__(self):
        self.hybrid_retrieval_service = HybridRetrievalService()
        self.reranking_service = RerankingService()

    def retrieve(
            self,
            request:VectorRetrievalRequest,
            db:Session
    )->list[HybridRetrievalResponse]:
        # Dense + BM25 + RRF，返回 rrf_top_k 条
        rrf_candidates=(
            self.hybrid_retrieval_service.retrieve(
                request=request,
                db=db
            )
        )

        if not rrf_candidates:
            return []

        # RRF Top 20 → Reranker → Final Top 5
        return self.reranking_service.rerank(
            query=request.query,
            candidates=rrf_candidates,
            top_k=request.top_k,
        )