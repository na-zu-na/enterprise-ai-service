import logging
import re
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.state import AgentRuntimeContext, AgentState
from agents.tools.get_time import get_time
from agents.tools.create_tasks import create_tasks
from agents.tools.query_tasks import query_tasks
from agents.tools.knowledge_search import knowledge_search
from agents.tools.approval import require_approval
from core.config import Settings
from agents.config.prompt import Prompt
from services.rag_service import (
    NO_RETRIEVAL_RESULT,
    NO_RETRIEVAL_RESULTS,
    get_rag_service,
)

logger = logging.getLogger("uvicorn.error")

SYSTEM_PROMPT =  Prompt.SYSTEM_PROMPT

BASE_TOOLS=[knowledge_search,query_tasks,create_tasks,get_time]

APPROVAL_REQUIRED_TOOL_NAMES = {
    "google_calendar_create_calendar_event",
}

FOLLOW_UP_ACTION_PATTERN = re.compile(
    r"\b(?:create|add|schedule|book|task|calendar|meeting|remind)\b"
    r"|创建|新增|安排|日程|任务|会议|提醒",
    re.IGNORECASE,
)

llm=ChatOpenAI(
    model=Settings.LLM_MODEL,
    api_key=Settings.LLM_API_KEY,
    base_url=Settings.LLM_BASE_URL,
    timeout=60,
    max_retries=2,
)

TITLE_SYSTEM_PROMPT = f"""
{Prompt.RESPONSE_LANGUAGE_RULE}

Generate a concise conversation title from the user's first message.
Requirements:
1. Accurately summarize the user's topic.
2. Use the same language as the user's first message.
3. Keep English titles between 3 and 10 words. Keep Chinese, Japanese, and Korean titles concise, normally between 6 and 20 characters.
4. Do not use quotation marks, terminal punctuation, or a "Title:" prefix.
5. Return only the title with no explanation.
""".strip()

DEFAULT_CONVERSATION_TITLE = Prompt.DEFAULT_CONVERSATION_TITLE


def extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return str(content)


def build_local_english_title(message: str) -> str | None:
    """Create an English title locally to avoid an extra LLM round trip."""

    if not message.isascii():
        return None

    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", message)
    if not words:
        return None

    selected_words: list[str] = []
    for word in words[:10]:
        candidate = " ".join([*selected_words, word])
        if len(candidate) > 50:
            break
        selected_words.append(word)

    title = " ".join(selected_words) or words[0][:50]
    return title[0].upper() + title[1:] if title else None


def generate_conversation_title(state: AgentState) -> str:
    first_user_message = next(
        (
            message
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    if first_user_message is None:
        return DEFAULT_CONVERSATION_TITLE

    first_message_text = extract_content_text(first_user_message.content).strip()
    fallback_title = DEFAULT_CONVERSATION_TITLE

    local_title = build_local_english_title(first_message_text)
    if local_title:
        return local_title

    try:
        started_at = perf_counter()
        response = llm.invoke(
            [
                SystemMessage(content=TITLE_SYSTEM_PROMPT),
                HumanMessage(content=first_message_text),
            ]
        )
        logger.info(
            "Conversation title generation completed in %.3f seconds",
            perf_counter() - started_at,
        )
    except Exception:
        logger.exception("Failed to generate conversation title")
        return fallback_title

    title = extract_content_text(response.content).strip()
    title = (
        title.removeprefix("Title:")
        .removeprefix("Title：")
        .removeprefix("标题：")
        .removeprefix("标题:")
    )
    title = title.splitlines()[0].strip(" #\"'“”‘’。") if title else ""
    if not title:
        return fallback_title
    return title[:50]


def get_trailing_tool_messages(state: AgentState) -> list[ToolMessage]:
    messages: list[ToolMessage] = []
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            break
        messages.append(message)
    return list(reversed(messages))


def get_latest_user_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return extract_content_text(message.content).strip()
    return ""


def route_after_tools(state: AgentState) -> str:
    tool_messages = get_trailing_tool_messages(state)
    if not tool_messages:
        return "agent"

    if all(
        message.name == "knowledge_search"
        and extract_content_text(message.content).strip()
        in NO_RETRIEVAL_RESULTS
        for message in tool_messages
    ):
        return "knowledge_answer"

    if (
        len(tool_messages) == 1
        and tool_messages[0].name == "knowledge_search"
        and not FOLLOW_UP_ACTION_PATTERN.search(get_latest_user_text(state))
    ):
        return "knowledge_answer"

    return "agent"



def build_graph(
        checkpointer: BaseCheckpointSaver,
        external_tools
)-> CompiledStateGraph[Any, Any, Any, Any]:
    guarded_external_tools = [
        require_approval(tool)
        if tool.name in APPROVAL_REQUIRED_TOOL_NAMES
        else tool
        for tool in external_tools
    ]

    tools=[
        *BASE_TOOLS,
        *guarded_external_tools,
    ]

    llm_with_tools=llm.bind_tools(tools)

    def agent_node(state:AgentState)->dict:
        started_at = perf_counter()
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )
        logger.info(
            "Agent LLM call completed in %.3f seconds",
            perf_counter() - started_at,
        )

        update = {
            "messages": [response]
        }

        if not state.get("title"):
            update["title"] = generate_conversation_title(state)

        return update

    def knowledge_answer_node(state: AgentState) -> dict:
        result = get_rag_service().generate_answer(
            get_latest_user_text(state),
            state.get("citations", []),
        )
        return {"messages": [AIMessage(content=result.answer)]}

    builder = StateGraph(
        AgentState,
        context_schema=AgentRuntimeContext,
    )
    # Agent 节点：调用模型
    builder.add_node(
        "agent",
        agent_node,
    )

    builder.add_node(
        "tools",
        ToolNode(tools)
    )

    builder.add_node(
        "knowledge_answer",
        knowledge_answer_node,
    )

    # 图的入口
    builder.add_edge(
        START,
        "agent",
    )

    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END:END
        }
    )

    # Simple knowledge answers use a compact RAG prompt. Empty retrieval
    # ends immediately; multi-step/tool requests continue through the agent.
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "agent": "agent",
            "knowledge_answer": "knowledge_answer",
            END: END,
        },
    )

    builder.add_edge("knowledge_answer", END)

    return builder.compile(checkpointer=checkpointer)
