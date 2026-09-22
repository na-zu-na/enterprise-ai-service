import logging
from time import perf_counter

from sqlalchemy.orm import Session

from core.config import Settings
from schemas.embedding import VectorRetrievalRequest, HybridRetrievalResponse
from services.citation_policy import filter_relevant_candidates
from services.hybrid_retrieval_service import HybridRetrievalService
from services.reranking_service import RerankingService


logger = logging.getLogger("uvicorn.error")


class RetrievalPipelineService:
    def __init__(self):
        self.hybrid_retrieval_service = HybridRetrievalService()
        self.reranking_service = RerankingService()

    def warm_up(self) -> None:
        if Settings.RERANKER_ENABLED:
            self.reranking_service.warm_up()

    def retrieve(
            self,
            request:VectorRetrievalRequest,
            db:Session
    )->list[HybridRetrievalResponse]:
        # Dense + BM25 + RRF，返回 rrf_top_k 条
        started_at = perf_counter()
        rrf_candidates=(
            self.hybrid_retrieval_service.retrieve(
                request=request,
                db=db
            )
        )
        logger.info(
            "Hybrid retrieval completed in %.3f seconds with %d candidates",
            perf_counter() - started_at,
            len(rrf_candidates),
        )

        if not rrf_candidates:
            return []

        if not Settings.RERANKER_ENABLED:
            logger.info(
                "Reranker disabled; returning RRF top %d",
                request.top_k,
            )
            return rrf_candidates[:request.top_k]

        # RRF Top 20 → Reranker → Final Top 5
        started_at = perf_counter()
        reranked_candidates = self.reranking_service.rerank(
            query=request.query,
            candidates=rrf_candidates,
            top_k=request.top_k,
        )
        logger.info(
            "Reranking completed in %.3f seconds for %d candidates",
            perf_counter() - started_at,
            len(rrf_candidates),
        )

        return filter_relevant_candidates(
            reranked_candidates,
            min_score=Settings.RERANKER_MIN_SCORE,
        )
