from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

from schemas.embedding import HybridRetrievalResponse


class AgentRuntimeContext(TypedDict):
    access_token: str


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    knowledge_base_ids: list[int]
    citations: list[HybridRetrievalResponse]
    title: NotRequired[str]
