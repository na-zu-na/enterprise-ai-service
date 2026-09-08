import unittest
from pathlib import Path
from types import SimpleNamespace

from evals.models import EvalCategory, RetrievalEvalCase
from evals.retrieval_eval import RetrievalEvaluator


class FakeDenseRetriever:
    def retrieve_dense(self, request, db):
        return [SimpleNamespace(id=14), SimpleNamespace(id=15)]


class FakeHybridService:
    def __init__(self):
        self.dense_retriever = FakeDenseRetriever()
        self.last_candidates = None

    def retrieve(self, request, db):
        self.last_candidates = [
            SimpleNamespace(id=15),
            SimpleNamespace(id=14),
        ]
        return self.last_candidates


class FakeRerankingService:
    def __init__(self, hybrid_service):
        self.hybrid_service = hybrid_service
        self.received_same_pool = False
        self.raise_on_error = False

    def rerank(self, *, query, candidates, top_k, raise_on_error=False):
        self.received_same_pool = candidates is self.hybrid_service.last_candidates
        self.raise_on_error = raise_on_error
        return list(reversed(candidates))[:top_k]


class RetrievalEvaluatorTest(unittest.TestCase):
    def test_three_strategies_and_reranker_reuses_rrf_pool(self):
        hybrid = FakeHybridService()
        reranker = FakeRerankingService(hybrid)
        evaluator = RetrievalEvaluator(
            hybrid_service=hybrid,
            reranking_service=reranker,
        )
        case = RetrievalEvalCase(
            id="case-1",
            category=EvalCategory.KEYWORD,
            query="年假需要提前几天申请？",
            knowledge_base_ids=[1],
            expected_document_id=1,
            expected_chunk_ids=[14],
        )

        report = evaluator.evaluate(
            cases=[case],
            db=SimpleNamespace(),
            dataset_path=Path("dataset.json"),
            candidate_k=2,
            rrf_top_k=2,
            top_k=2,
        )

        self.assertEqual(len(report.strategies), 3)
        self.assertTrue(reranker.received_same_pool)
        self.assertTrue(reranker.raise_on_error)
        self.assertEqual(report.strategies[0].queries[0].retrieved_chunk_ids, [14, 15])
        self.assertEqual(report.strategies[1].queries[0].retrieved_chunk_ids, [15, 14])
        self.assertEqual(report.strategies[2].queries[0].retrieved_chunk_ids, [14, 15])


if __name__ == "__main__":
    unittest.main()

