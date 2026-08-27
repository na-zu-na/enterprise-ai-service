import json
from pathlib import Path

from evals.models import RetrievalEvalReport, StrategyEvalResult


def print_console_report(report: RetrievalEvalReport) -> None:
    for strategy in report.strategies:
        _print_strategy(strategy)

    if len(report.strategies) >= 3:
        hybrid = report.strategies[1].metrics
        reranked = report.strategies[2].metrics
        print("\n" + "=" * 30)
        print("Reranker vs Hybrid + RRF")
        print("=" * 30)
        print(f"Hit@{report.top_k}:    {reranked.hit_at_k - hybrid.hit_at_k:+.4f}")
        print(
            f"Recall@{report.top_k}: "
            f"{reranked.recall_at_k - hybrid.recall_at_k:+.4f}"
        )
        print(f"MRR:      {reranked.mrr - hybrid.mrr:+.4f}")


def write_json_report(report: RetrievalEvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _print_strategy(result: StrategyEvalResult) -> None:
    metrics = result.metrics
    print("\n" + "=" * 30)
    print(result.strategy)
    print("=" * 30)
    print(f"Evaluated: {metrics.evaluated_queries}")
    if metrics.skipped_no_answer_queries:
        print(f"No-answer diagnostics: {metrics.skipped_no_answer_queries}")
    print(f"Hit@{result.top_k}:    {metrics.hit_at_k:.4f}")
    print(f"Recall@{result.top_k}: {metrics.recall_at_k:.4f}")
    print(f"MRR:      {metrics.mrr:.4f}")

    if result.metrics_by_category:
        print("By category:")
        for category, category_metrics in result.metrics_by_category.items():
            if category_metrics.evaluated_queries == 0:
                continue
            print(
                f"  {category}: hit={category_metrics.hit_at_k:.4f}, "
                f"recall={category_metrics.recall_at_k:.4f}, "
                f"mrr={category_metrics.mrr:.4f}"
            )

