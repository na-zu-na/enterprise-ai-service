import logging
from threading import Lock

from rerankers.bge_reranker import BgeReranker
from schemas.embedding import HybridRetrievalResponse

logger = logging.getLogger(__name__)


class RerankingService:
    def __init__(self) -> None:
        self._reranker: BgeReranker | None = None
        self._load_lock = Lock()

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

    def _get_reranker(self) -> BgeReranker:
        if self._reranker is None:
            with self._load_lock:
                if self._reranker is None:
                    self._reranker = BgeReranker()

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
