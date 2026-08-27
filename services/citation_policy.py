import re

from schemas.embedding import HybridRetrievalResponse


_INSUFFICIENT_ANSWER_PATTERN = re.compile(
    r"^(?:抱歉[，,。.!！\s]*)?根据现有资料无法确定[。.!！\s]*$"
)


def is_insufficient_answer(answer: str) -> bool:
    """判断答案是否为知识库拒答。"""

    normalized_answer = " ".join(answer.strip().split())
    return bool(_INSUFFICIENT_ANSWER_PATTERN.fullmatch(normalized_answer))


def select_answer_citations(
        answer: str,
        candidates: list[HybridRetrievalResponse],
) -> list[HybridRetrievalResponse]:
    """只返回最终答案中明确引用的候选片段。"""

    if not candidates or is_insufficient_answer(answer):
        return []

    cited_document_names = {
        candidate.document_name
        for candidate in candidates
        if f"【{candidate.document_name}】" in answer
    }

    if not cited_document_names:
        return []

    return [
        candidate
        for candidate in candidates
        if candidate.document_name in cited_document_names
    ]


def filter_relevant_candidates(
        candidates: list[HybridRetrievalResponse],
        *,
        min_score: float,
) -> list[HybridRetrievalResponse]:
    """过滤低相关候选；reranker 降级时保留 RRF 结果。"""

    return [
        candidate
        for candidate in candidates
        if (
            candidate.rerank_score is None
            or candidate.rerank_score >= min_score
        )
    ]
