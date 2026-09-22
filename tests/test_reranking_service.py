import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep

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


class ConcurrencyDetectingReranker:
    def __init__(self):
        self._active_call = Lock()

    def compute_scores(self, query, passages):
        if not self._active_call.acquire(blocking=False):
            raise RuntimeError("concurrent reranker inference")
        try:
            sleep(0.02)
            return [0.9] * len(passages)
        finally:
            self._active_call.release()


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

    def test_serializes_concurrent_inference_on_shared_reranker(self):
        service = RerankingService()
        service._reranker = ConcurrencyDetectingReranker()
        workers = 4
        start = Barrier(workers)

        def rerank_once(chunk_id):
            start.wait()
            return service.rerank(
                query="query",
                candidates=[candidate(chunk_id, 0.9)],
                top_k=1,
                raise_on_error=True,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(rerank_once, chunk_id)
                for chunk_id in range(1, workers + 1)
            ]
            results = [future.result() for future in futures]

        self.assertEqual(
            [rows[0].id for rows in results],
            list(range(1, workers + 1)),
        )


if __name__ == "__main__":
    unittest.main()
