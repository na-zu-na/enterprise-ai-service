from pydantic import BaseModel,Field

from schemas.embedding import HybridRetrievalResponse


class AgentChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        description="用户发送的消息"
    )

    knowledge_base_ids: list[int] = Field(
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

class AgentChatResponse(BaseModel):
    answer: str
    title: str
    citations: list[HybridRetrievalResponse]
    checkpoint: CheckpointRef
