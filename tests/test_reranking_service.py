import unittest

from schemas.embedding import HybridRetrievalResponse
from services.reranking_service import RerankingService


def candidate(chunk_id, rrf_score):
    return HybridRetrievalResponse(
        id=chunk_id,
        document_id=1,
        knowledge_base_id=1,
        chunk_index=chunk_id,
        content=f"内容 {chunk_id}",
        document_name="年假管理制度",
        section_title="年假申请",
        rrf_score=rrf_score,
    )


class StubReranker:
    def __init__(self, scores=None, error=None):
        self.scores = scores
        self.error = error

    def compute_scores(self, query, passages):
        if self.error:
            raise self.error
        return self.scores


class RerankingServiceTest(unittest.TestCase):
    def test_reranks_and_assigns_ranks(self):
        service = RerankingService()
        service._reranker = StubReranker([0.1, 0.9])
        result = service.rerank(
            query="query",
            candidates=[candidate(14, 0.9), candidate(15, 0.8)],
            top_k=2,
        )
        self.assertEqual([row.id for row in result], [15, 14])
        self.assertEqual([row.rerank_rank for row in result], [1, 2])

    def test_production_mode_falls_back_to_rrf_order(self):
        service = RerankingService()
        service._reranker = StubReranker(error=ValueError("failed"))
        candidates = [candidate(14, 0.9), candidate(15, 0.8)]
        result = service.rerank(
            query="query",
            candidates=candidates,
            top_k=1,
        )
        self.assertEqual([row.id for row in result], [14])

    def test_evaluation_mode_exposes_reranker_failure(self):
        service = RerankingService()
        service._reranker = StubReranker(error=ValueError("failed"))
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            service.rerank(
                query="query",
                candidates=[candidate(14, 0.9)],
                top_k=1,
                raise_on_error=True,
            )

    def test_passage_uses_readable_labels(self):
        passage = RerankingService._build_passage(candidate(14, 0.9))
        self.assertIn("文档名称：年假管理制度", passage)
        self.assertIn("章节：年假申请", passage)
        self.assertIn("内容：内容 14", passage)


if __name__ == "__main__":
    unittest.main()

