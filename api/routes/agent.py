from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command, Overwrite
from langgraph.graph.state import CompiledStateGraph

from clients.tools.get_access_token import get_access_token
from core.ApiResponse import ApiResponse
from schemas.agent import AgentChatRequest, AgentChatResponse, CheckpointRef, ApprovalRequest, \
    AgentApprovalDecisionRequest
from services.citation_policy import select_answer_citations
from services.knowledge_access_service import get_accessible_knowledge_base_ids

router=APIRouter(
    prefix="/agent",
    tags=["agent"],
)

#会话配置构造函数
def get_authenticated_user_id(
        user_id: Annotated[
            int | None,
            Header(alias="X-Authenticated-User-Id"),
        ] = None,
) -> int:
    if user_id is None or user_id <= 0:
        raise HTTPException(
            status_code=401,
            detail="缺少有效的已认证用户 ID",
        )
    return user_id


def build_graph_config(conversation_id: int, user_id: int)->dict:
    return {
        "configurable": {
            "thread_id": f"user:{user_id}:conversation:{conversation_id}",
        }
    }

#增加checkpoint构造函数
async def build_checkpoint_ref(
        graph: CompiledStateGraph,
        config: dict,
) -> CheckpointRef:
    snapshot = await graph.aget_state(config)

    snapshot_config = snapshot.config["configurable"]

    parent_config = (
        snapshot.parent_config.get("configurable", {})
        if snapshot.parent_config
        else {}
    )

    return CheckpointRef(
        checkpoint_id=snapshot_config["checkpoint_id"],
        checkpoint_ns=snapshot_config.get("checkpoint_ns", ""),
        parent_checkpoint_id=parent_config.get("checkpoint_id"),
        created_at=snapshot.created_at,
    )

async def build_agent_response(
        graph: CompiledStateGraph,
        config: dict,
        result: dict[str, Any],
)->AgentChatResponse:
    checkpoint=await build_checkpoint_ref(
        graph=graph,
        config=config,
    )

    interrupts=result.get("__interrupt__",())

    if interrupts:
        current_interrupt=interrupts[0]
        interrupt_value=current_interrupt.value
        if not isinstance(interrupt_value, dict):
            interrupt_value={
                "action":"unknown",
                "payload":{}
            }

        approval_payload = interrupt_value.get("payload", {})
        if not isinstance(approval_payload, dict):
            approval_payload = {}

        return AgentChatResponse(
            status="approval_required",
            answer=None,
            approval=ApprovalRequest(
                interrupt_id=current_interrupt.id,
                action=str(
                    interrupt_value.get("action", "unknown")
                ),
                payload=approval_payload
            ),
            title=str(result.get("title") or "新对话"),
            citations=result.get("citations", []),
            checkpoint=checkpoint,
        )

    final_message = result["messages"][-1]
    answer = extract_message_text(final_message.content)

    citations = select_answer_citations(
        answer,
        result.get("citations", []),
    )

    return AgentChatResponse(
        status="completed",
        answer=answer,
        approval=None,
        title=str(result.get("title") or "新对话"),
        citations=citations,
        checkpoint=checkpoint,
    )

@router.post(
    "/chat",
    response_model=ApiResponse[AgentChatResponse]
)
async def agent_chat(
        request: Request,
        payload: AgentChatRequest,
        access_token:Annotated[
            str,
            Depends(get_access_token),
        ],
        user_id: int = Depends(get_authenticated_user_id),
        accessible_ids: list[int] = Depends(
            get_accessible_knowledge_base_ids
        ),
)->ApiResponse[AgentChatResponse]:
    graph: CompiledStateGraph = request.app.state.agent_graph
    if not set(payload.knowledge_base_ids).issubset(accessible_ids):
        raise HTTPException(
            status_code=403,
            detail="请求包含无权访问的知识库",
        )

    config = build_graph_config(payload.conversation_id, user_id)

    result=await graph.ainvoke(
        {
            "messages":[
                HumanMessage(content=payload.message)
            ],
            "knowledge_base_ids":payload.knowledge_base_ids,
            # citations 使用 reducer 合并并行检索；每个新用户回合先清空旧引用。
            "citations": Overwrite(value=[]),
        },
        config=config,
        context={
            "access_token": access_token,
        },
    )

    response=await build_agent_response(
        graph=graph,
        config=config,
        result=result,
    )

    return ApiResponse[AgentChatResponse].success(response)

@router.post(
    "/approvals/respond",
    response_model=ApiResponse[AgentChatResponse],
)
async def respond_to_approval(
        request: Request,
        payload: AgentApprovalDecisionRequest,
        access_token: Annotated[
            str,
            Depends(get_access_token),
        ],
        user_id: int = Depends(get_authenticated_user_id),
) -> ApiResponse[AgentChatResponse]:
    graph: CompiledStateGraph = request.app.state.agent_graph
    config = build_graph_config(payload.conversation_id, user_id)

    snapshot=await graph.aget_state(config)

    if not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail="没有找到对应的会话状态",
    )

    pending_interrupt_ids = {
        interrupt.id
        for task in snapshot.tasks
        for interrupt in getattr(task, "interrupts", ())
    }

    if payload.interrupt_id not in pending_interrupt_ids:
        raise HTTPException(
            status_code=409,
            detail="审批请求不存在、已经处理或不属于当前会话",
        )

    result = await graph.ainvoke(
        Command(
            resume={
                payload.interrupt_id: {
                    "approved": payload.approved,
                }
            }
        ),
        config=config,
        context={
            "access_token": access_token,
        },
    )

    response = await build_agent_response(
        graph=graph,
        config=config,
        result=result,
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
