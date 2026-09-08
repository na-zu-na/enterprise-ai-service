import unittest

from evals.metrics import aggregate_metrics, calculate_query_metrics


class QueryMetricsTest(unittest.TestCase):
    def test_full_recall_and_first_rank(self):
        result = calculate_query_metrics(
            expected_chunk_ids=[14, 15],
            retrieved_chunk_ids=[14, 13, 16, 15],
            top_k=5,
        )
        self.assertEqual(result.hit_at_k, 1.0)
        self.assertEqual(result.recall_at_k, 1.0)
        self.assertEqual(result.reciprocal_rank, 1.0)

    def test_partial_recall_and_reciprocal_rank(self):
        result = calculate_query_metrics(
            expected_chunk_ids=[14, 15],
            retrieved_chunk_ids=[13, 14, 16],
            top_k=3,
        )
        self.assertEqual(result.hit_at_k, 1.0)
        self.assertEqual(result.recall_at_k, 0.5)
        self.assertEqual(result.reciprocal_rank, 0.5)

    def test_miss(self):
        result = calculate_query_metrics(
            expected_chunk_ids=[14],
            retrieved_chunk_ids=[13, 15, 16],
            top_k=3,
        )
        self.assertEqual(result.hit_at_k, 0.0)
        self.assertEqual(result.recall_at_k, 0.0)
        self.assertEqual(result.reciprocal_rank, 0.0)

    def test_top_k_is_respected(self):
        result = calculate_query_metrics(
            expected_chunk_ids=[14],
            retrieved_chunk_ids=[13, 15, 14],
            top_k=2,
        )
        self.assertEqual(result.hit_at_k, 0.0)

    def test_no_answer_is_not_scored(self):
        result = calculate_query_metrics(
            expected_chunk_ids=[],
            retrieved_chunk_ids=[13, 14],
            top_k=5,
        )
        self.assertIsNone(result.hit_at_k)
        self.assertIsNone(result.recall_at_k)
        self.assertIsNone(result.reciprocal_rank)

    def test_macro_average_skips_no_answer(self):
        hit = calculate_query_metrics(
            expected_chunk_ids=[14],
            retrieved_chunk_ids=[14],
            top_k=5,
        )
        miss = calculate_query_metrics(
            expected_chunk_ids=[15],
            retrieved_chunk_ids=[14],
            top_k=5,
        )
        no_answer = calculate_query_metrics(
            expected_chunk_ids=[],
            retrieved_chunk_ids=[14],
            top_k=5,
        )
        result = aggregate_metrics([hit, miss, no_answer])
        self.assertEqual(result.evaluated_queries, 2)
        self.assertEqual(result.skipped_no_answer_queries, 1)
        self.assertEqual(result.hit_at_k, 0.5)
        self.assertEqual(result.recall_at_k, 0.5)
        self.assertEqual(result.mrr, 0.5)


if __name__ == "__main__":
    unittest.main()

