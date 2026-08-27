from sqlalchemy.orm import Session

from core.config import Settings
from schemas.embedding import VectorRetrievalRequest, HybridRetrievalResponse
from services.citation_policy import filter_relevant_candidates
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
        reranked_candidates = self.reranking_service.rerank(
            query=request.query,
            candidates=rrf_candidates,
            top_k=request.top_k,
        )

        return filter_relevant_candidates(
            reranked_candidates,
            min_score=Settings.RERANKER_MIN_SCORE,
        )
