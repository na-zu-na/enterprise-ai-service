from typing import Annotated

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from clients.tools.get_access_token import get_access_token
from core.ApiResponse import ApiResponse
from schemas.agent import AgentChatRequest, AgentChatResponse, CheckpointRef
from services.citation_policy import select_answer_citations

router=APIRouter(
    prefix="/agent",
    tags=["agent"],
)

@router.post(
    "/chat",
    response_model=ApiResponse[AgentChatResponse]
)
def agent_chat(
        request: Request,
        payload: AgentChatRequest,
        access_token:Annotated[
            str,
            Depends(get_access_token),
        ]
)->ApiResponse[AgentChatResponse]:
    graph: CompiledStateGraph = request.app.state.agent_graph
    config = {
        "configurable": {
            # conversation_id 必须全局唯一
            "thread_id": f"conversation:{payload.conversation_id}",
        }
    }

    result=graph.invoke(
        {
            "messages":[
                HumanMessage(content=payload.message)
            ],
            "knowledge_base_ids":payload.knowledge_base_ids,
            "citations":[],
        },
        config=config,
        context={
            "access_token": access_token,
        },
    )

    final_message=result["messages"][-1]
    answer=extract_message_text(final_message.content)
    citations=select_answer_citations(
        answer,
        result.get("citations", []),
    )

    snapshot = graph.get_state(config)
    snapshot_config = snapshot.config["configurable"]
    parent_config = (
        snapshot.parent_config.get("configurable", {})
        if snapshot.parent_config
        else {}
    )

    response = AgentChatResponse(
        answer=answer,
        title=result["title"],
        citations=citations,
        checkpoint=CheckpointRef(
            checkpoint_id=snapshot_config["checkpoint_id"],
            checkpoint_ns=snapshot_config.get("checkpoint_ns", ""),
            parent_checkpoint_id=parent_config.get("checkpoint_id"),
            created_at=snapshot.created_at,
        ),
    )

    return ApiResponse[AgentChatResponse].success(response)


def extract_message_text(content) -> str:
    """兼容普通字符串和 Responses API 内容块。"""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return str(content)
