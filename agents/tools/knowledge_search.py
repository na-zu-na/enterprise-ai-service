from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from db.session import SessionLocal
from schemas.embedding import VectorRetrievalRequest
from services.rag_service import RagService

rag_service=RagService()

@tool
def knowledge_search(
        query: str,
        knowledge_base_ids:Annotated[
            list[int],
            InjectedState("knowledge_base_ids")
        ],
        tool_call_id: Annotated[
            str,
            InjectedToolCallId
        ]
)->Command:
    """在企业知识库中搜索资料
    当用户的问题需要查询企业内部文档或知识库时调用。
    """
    request=VectorRetrievalRequest(
        query=query,
        knowledge_base_ids=knowledge_base_ids
    )

    with SessionLocal() as db:
        chunks=rag_service.retrieve(
            request=request,
            db=db,
        )

    if chunks:
        tool_content = RagService.build_context(chunks)
    else:
        tool_content = "没有检索到相关知识库内容。"

    return Command(
        update={
            "citations": chunks,
            "messages": [
                ToolMessage(
                    name="knowledge_search",
                    content=tool_content,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
