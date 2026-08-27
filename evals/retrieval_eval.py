import argparse
import json
from collections import defaultdict
from pathlib import Path

from pydantic import TypeAdapter
from sqlalchemy.orm import Session

from db.session import SessionLocal
from evals.metrics import aggregate_metrics, calculate_query_metrics
from evals.models import (
    QueryEvalResult,
    RetrievalEvalCase,
    RetrievalEvalReport,
    StrategyEvalResult,
)
from evals.reporters import print_console_report, write_json_report
from models.document_chunk import DocumentChunkEntity
from schemas.embedding import VectorRetrievalRequest
from services.hybrid_retrieval_service import HybridRetrievalService
from services.reranking_service import RerankingService


DEFAULT_DATASET = Path(__file__).parent / "datasets" / "retrieval_eval.json"


def load_dataset(path: Path) -> list[RetrievalEvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = TypeAdapter(list[RetrievalEvalCase]).validate_python(raw)

    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case ids must be unique")
    if not cases:
        raise ValueError("evaluation dataset must not be empty")
    return cases


def validate_dataset_against_db(
        cases: list[RetrievalEvalCase],
        db: Session,
) -> None:
    expected_ids = {
        chunk_id
        for case in cases
        for chunk_id in case.expected_chunk_ids
    }
    if not expected_ids:
        return

    rows = db.query(DocumentChunkEntity).filter(
        DocumentChunkEntity.id.in_(expected_ids)
    ).all()
    chunks = {row.id: row for row in rows}
    missing = sorted(expected_ids - chunks.keys())
    if missing:
        raise ValueError(f"expected chunks do not exist: {missing}")

    for case in cases:
        for chunk_id in case.expected_chunk_ids:
            chunk = chunks[chunk_id]
            if chunk.document_id != case.expected_document_id:
                raise ValueError(
                    f"case {case.id}: chunk {chunk_id} belongs to document "
                    f"{chunk.document_id}, not {case.expected_document_id}"
                )
            if chunk.knowledge_base_id not in case.knowledge_base_ids:
                raise ValueError(
                    f"case {case.id}: chunk {chunk_id} is outside the "
                    "configured knowledge_base_ids"
                )


class RetrievalEvaluator:
    def __init__(
            self,
            *,
            hybrid_service: HybridRetrievalService | None = None,
            reranking_service: RerankingService | None = None,
    ) -> None:
        self.hybrid_service = hybrid_service or HybridRetrievalService()
        self.reranking_service = reranking_service or RerankingService()

    def evaluate(
            self,
            *,
            cases: list[RetrievalEvalCase],
            db: Session,
            dataset_path: Path,
            candidate_k: int = 30,
            rrf_top_k: int = 20,
            top_k: int = 5,
    ) -> RetrievalEvalReport:
        strategy_queries: dict[str, list[QueryEvalResult]] = {
            "Vector Only": [],
            "Hybrid + RRF": [],
            "Hybrid + RRF + Reranker": [],
        }

        for case in cases:
            request = VectorRetrievalRequest(
                query=case.query,
                knowledge_base_ids=case.knowledge_base_ids,
                candidate_k=candidate_k,
                rrf_top_k=rrf_top_k,
                top_k=top_k,
            )

            dense_rows = self.hybrid_service.dense_retriever.retrieve_dense(
                request,
                db,
            )
            rrf_candidates = self.hybrid_service.retrieve(request, db)
            # C deliberately consumes the exact candidate pool produced by B.
            reranked_candidates = self.reranking_service.rerank(
                query=case.query,
                candidates=rrf_candidates,
                top_k=top_k,
                raise_on_error=True,
            )

            retrieved_by_strategy = {
                "Vector Only": [row.id for row in dense_rows[:top_k]],
                "Hybrid + RRF": [row.id for row in rrf_candidates[:top_k]],
                "Hybrid + RRF + Reranker": [
                    row.id for row in reranked_candidates
                ],
            }

            for strategy, retrieved_ids in retrieved_by_strategy.items():
                strategy_queries[strategy].append(
                    QueryEvalResult(
                        case_id=case.id,
                        category=case.category,
                        query=case.query,
                        expected_chunk_ids=case.expected_chunk_ids,
                        retrieved_chunk_ids=retrieved_ids,
                        metrics=calculate_query_metrics(
                            expected_chunk_ids=case.expected_chunk_ids,
                            retrieved_chunk_ids=retrieved_ids,
                            top_k=top_k,
                        ),
                    )
                )

        strategies = [
            self._summarize_strategy(name, queries, top_k)
            for name, queries in strategy_queries.items()
        ]
        return RetrievalEvalReport(
            dataset_path=str(dataset_path),
            candidate_k=candidate_k,
            rrf_top_k=rrf_top_k,
            top_k=top_k,
            strategies=strategies,
        )

    @staticmethod
    def _summarize_strategy(
            name: str,
            queries: list[QueryEvalResult],
            top_k: int,
    ) -> StrategyEvalResult:
        grouped = defaultdict(list)
        for query in queries:
            grouped[query.category.value].append(query.metrics)

        return StrategyEvalResult(
            strategy=name,
            top_k=top_k,
            metrics=aggregate_metrics(query.metrics for query in queries),
            metrics_by_category={
                category: aggregate_metrics(category_metrics)
                for category, category_metrics in sorted(grouped.items())
            },
            queries=queries,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Vector, Hybrid RRF, and Reranker retrieval.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-top-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results/retrieval_eval_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_dataset(args.dataset)
    evaluator = RetrievalEvaluator()

    with SessionLocal() as db:
        validate_dataset_against_db(cases, db)
        report = evaluator.evaluate(
            cases=cases,
            db=db,
            dataset_path=args.dataset,
            candidate_k=args.candidate_k,
            rrf_top_k=args.rrf_top_k,
            top_k=args.top_k,
        )

    print_console_report(report)
    write_json_report(report, args.output)
    print(f"\nDetailed report: {args.output}")


if __name__ == "__main__":
    main()
