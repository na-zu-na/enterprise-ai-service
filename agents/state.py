from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

from schemas.embedding import HybridRetrievalResponse


def merge_citations(
        current: list[HybridRetrievalResponse],
        new: list[HybridRetrievalResponse],
) -> list[HybridRetrievalResponse]:
    """合并并行知识库检索结果，并按 Chunk ID 去重。"""
    merged: list[HybridRetrievalResponse] = []
    seen_chunk_ids: set[int] = set()

    for citation in [*(current or []), *(new or [])]:
        if citation.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(citation.id)
        merged.append(citation)

    return merged


class AgentRuntimeContext(TypedDict):
    access_token: str


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    knowledge_base_ids: list[int]
    citations: Annotated[
        list[HybridRetrievalResponse],
        merge_citations,
    ]
    title: NotRequired[str]
