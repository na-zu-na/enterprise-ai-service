from typing import Annotated, Any, Literal

from pydantic import BaseModel,Field

from schemas.embedding import HybridRetrievalResponse


class AgentChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        description="用户发送的消息"
    )

    knowledge_base_ids: list[Annotated[int, Field(gt=0)]] = Field(
        min_length=1,
        max_length=200,
        description="本次对话允许查询的知识库 ID"
    )

    conversation_id: int = Field(
        gt=0,
        description="本次会话的id"
    )

class CheckpointRef(BaseModel):
    checkpoint_id: str
    checkpoint_ns: str = ""
    parent_checkpoint_id: str | None = None
    created_at: str | None = None

class ApprovalRequest(BaseModel):
    interrupt_id: str
    action: str
    payload: dict[str, Any]

class AgentApprovalDecisionRequest(BaseModel):
    conversation_id: int = Field(
        gt=0,
        description="需要恢复的会话 ID",
    )

    interrupt_id: str = Field(
        min_length=1,
        description="LangGraph 中断 ID",
    )

    approved: bool = Field(
        description="是否允许执行操作",
    )

class AgentChatResponse(BaseModel):
    status: Literal["completed", "approval_required"]
    answer: str | None = None
    approval: ApprovalRequest | None = None
    title: str
    citations: list[HybridRetrievalResponse]
    checkpoint: CheckpointRef
