from sqlalchemy.orm import Session

from retrieval.bm25_retriever import Bm25Retriever
from retrieval.models import RetrievalCandidate
from retrieval.rrf_fusion_service import RrfFusionService
from schemas.embedding import (
    HybridRetrievalResponse,
    VectorRetrievalRequest,
)
from services.embedding_service import EmbeddingService


class HybridRetrievalService:
    def __init__(self) -> None:
        self.dense_retriever = EmbeddingService()
        self.bm25_retriever = Bm25Retriever()
        self.rrf_fusion = RrfFusionService()

    def retrieve(
            self,
            request: VectorRetrievalRequest,
            db: Session,
    ) -> list[HybridRetrievalResponse]:
        dense_rows = self.dense_retriever.retrieve_dense(
            request,
            db,
        )

        dense_candidates = [
            RetrievalCandidate(
                id=row.id,
                document_id=row.document_id,
                knowledge_base_id=row.knowledge_base_id,
                chunk_index=row.chunk_index,
                content=row.content,
                document_name=row.document_name,
                section_title=row.section_title,
                dense_distance=row.distance,
            )
            for row in dense_rows
        ]

        bm25_candidates = self.bm25_retriever.retrieve(request)

        fused_candidates = self.rrf_fusion.fuse(
            dense_hits=dense_candidates,
            bm25_hits=bm25_candidates,
            top_k=request.rrf_top_k,
            rank_constant=60,
        )

        return [
            HybridRetrievalResponse(
                id=hit.id,
                document_id=hit.document_id,
                knowledge_base_id=hit.knowledge_base_id,
                chunk_index=hit.chunk_index,
                content=hit.content,
                document_name=hit.document_name,
                section_title=hit.section_title,
                rrf_score=hit.rrf_score,
                dense_rank=hit.dense_rank,
                dense_distance=hit.dense_distance,
                bm25_rank=hit.bm25_rank,
                bm25_score=hit.bm25_score,
            )
            for hit in fused_candidates
        ]
