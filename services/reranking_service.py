import logging
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING

from schemas.embedding import HybridRetrievalResponse

if TYPE_CHECKING:
    from rerankers.bge_reranker import BgeReranker

logger = logging.getLogger("uvicorn.error")


class RerankingService:
    def __init__(self) -> None:
        self._reranker: "BgeReranker | None" = None
        self._load_error: Exception | None = None
        self._load_lock = Lock()
        # Hugging Face fast tokenizers mutate truncation/padding state during
        # encoding and cannot safely be borrowed by multiple threads at once.
        self._inference_lock = Lock()

    def warm_up(self) -> None:
        """Load the reranker before serving latency-sensitive requests."""

        started_at = perf_counter()
        reranker = self._get_reranker()
        with self._inference_lock:
            reranker.compute_scores(
                "warm-up query",
                ["warm-up passage"],
            )
        logger.info(
            "Reranker warm-up completed in %.3f seconds",
            perf_counter() - started_at,
        )

    def rerank(
            self,
            *,
            query: str,
            candidates: list[HybridRetrievalResponse],
            top_k: int,
            raise_on_error: bool = False,
    ) -> list[HybridRetrievalResponse]:
        if not candidates:
            return []

        passages = [
            self._build_passage(candidate)
            for candidate in candidates
        ]

        try:
            reranker = self._get_reranker()
            with self._inference_lock:
                scores = reranker.compute_scores(query, passages)
            if len(scores) != len(candidates):
                raise RuntimeError(
                    "Reranker score count does not match candidate count"
                )
        except Exception as exc:
            logger.exception(
                "Reranker failed; falling back to RRF ranking"
            )
            if raise_on_error:
                raise RuntimeError("Reranker evaluation failed") from exc
            return candidates[:top_k]

        scored_candidates = [
            candidate.model_copy(update={"rerank_score": score})
            for candidate, score in zip(candidates, scores, strict=True)
        ]

        # Reranker 分数是主要排序依据，RRF 分数用于处理同分情况。
        scored_candidates.sort(
            key=lambda candidate: (
                candidate.rerank_score,
                candidate.rrf_score,
            ),
            reverse=True,
        )

        return [
            candidate.model_copy(update={"rerank_rank": rank})
            for rank, candidate in enumerate(
                scored_candidates[:top_k],
                start=1,
            )
        ]

    def _get_reranker(self) -> "BgeReranker":
        if self._load_error is not None:
            raise RuntimeError(
                "Reranker initialization previously failed"
            ) from self._load_error

        if self._reranker is None:
            with self._load_lock:
                if self._reranker is None:
                    from rerankers.bge_reranker import BgeReranker

                    try:
                        self._reranker = BgeReranker()
                    except Exception as exc:
                        self._load_error = exc
                        raise

        if self._reranker is None:
            raise RuntimeError("Reranker initialization failed")
        return self._reranker

    @staticmethod
    def _build_passage(candidate: HybridRetrievalResponse) -> str:
        parts = []
        if candidate.document_name:
            parts.append(f"文档名称：{candidate.document_name}")
        if candidate.section_title:
            parts.append(f"章节：{candidate.section_title}")
        parts.append(f"内容：{candidate.content}")
        return "\n".join(parts)
