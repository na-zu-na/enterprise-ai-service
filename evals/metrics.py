from collections.abc import Iterable, Sequence

from evals.models import AggregateMetrics, QueryMetrics


def calculate_query_metrics(
        *,
        expected_chunk_ids: Sequence[int],
        retrieved_chunk_ids: Sequence[int],
        top_k: int,
) -> QueryMetrics:
    """Calculate Hit@K, Recall@K and reciprocal rank for one query."""
    if top_k < 1:
        raise ValueError("top_k must be greater than 0")

    # No-answer rows are diagnostic only: these metrics are undefined when
    # there is no relevant chunk, so they are excluded from macro averages.
    if not expected_chunk_ids:
        return QueryMetrics(
            hit_at_k=None,
            recall_at_k=None,
            reciprocal_rank=None,
        )

    expected = set(expected_chunk_ids)
    top_results = list(retrieved_chunk_ids[:top_k])
    matched = expected.intersection(top_results)

    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(top_results, start=1):
        if chunk_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    return QueryMetrics(
        hit_at_k=float(bool(matched)),
        recall_at_k=len(matched) / len(expected),
        reciprocal_rank=reciprocal_rank,
    )


def aggregate_metrics(metrics: Iterable[QueryMetrics]) -> AggregateMetrics:
    rows = list(metrics)
    scored = [row for row in rows if row.hit_at_k is not None]
    skipped = len(rows) - len(scored)

    if not scored:
        return AggregateMetrics(
            evaluated_queries=0,
            skipped_no_answer_queries=skipped,
            hit_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
        )

    count = len(scored)
    return AggregateMetrics(
        evaluated_queries=count,
        skipped_no_answer_queries=skipped,
        hit_at_k=sum(row.hit_at_k or 0.0 for row in scored) / count,
        recall_at_k=sum(row.recall_at_k or 0.0 for row in scored) / count,
        mrr=sum(row.reciprocal_rank or 0.0 for row in scored) / count,
    )

