from schemas.embedding import HybridRetrievalResponse
from services.citation_policy import (
    filter_relevant_candidates,
    is_insufficient_answer,
    select_answer_citations,
)


def make_candidate(
        *,
        chunk_id: int,
        document_name: str,
        rerank_score: float | None,
) -> HybridRetrievalResponse:
    return HybridRetrievalResponse(
        id=chunk_id,
        document_id=chunk_id,
        knowledge_base_id=1,
        chunk_index=0,
        content="测试内容",
        document_name=document_name,
        section_title="测试章节",
        rrf_score=0.1,
        rerank_score=rerank_score,
    )


def test_insufficient_answer_has_no_citations() -> None:
    candidate = make_candidate(
        chunk_id=1,
        document_name="制度文档",
        rerank_score=0.9,
    )

    answer = "根据现有资料无法确定。"

    assert is_insufficient_answer(answer)
    assert select_answer_citations(answer, [candidate]) == []


def test_only_explicitly_cited_documents_are_returned() -> None:
    cited = make_candidate(
        chunk_id=1,
        document_name="制度文档",
        rerank_score=0.9,
    )
    not_cited = make_candidate(
        chunk_id=2,
        document_name="操作手册",
        rerank_score=0.8,
    )

    citations = select_answer_citations(
        "报销期限为30天【制度文档】。",
        [cited, not_cited],
    )

    assert citations == [cited]


def test_answer_without_document_reference_has_no_citations() -> None:
    candidate = make_candidate(
        chunk_id=1,
        document_name="制度文档",
        rerank_score=0.9,
    )

    assert select_answer_citations("报销期限为30天。", [candidate]) == []


def test_rerank_score_threshold_filters_low_relevance() -> None:
    low_score = make_candidate(
        chunk_id=1,
        document_name="低相关文档",
        rerank_score=0.49,
    )
    accepted_score = make_candidate(
        chunk_id=2,
        document_name="相关文档",
        rerank_score=0.5,
    )
    rrf_fallback = make_candidate(
        chunk_id=3,
        document_name="降级结果",
        rerank_score=None,
    )

    result = filter_relevant_candidates(
        [low_score, accepted_score, rrf_fallback],
        min_score=0.5,
    )

    assert result == [accepted_score, rrf_fallback]
