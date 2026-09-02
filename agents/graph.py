import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
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

logger = logging.getLogger(__name__)

SYSTEM_PROMPT =  Prompt.SYSTEM_PROMPT

BASE_TOOLS=[knowledge_search,query_tasks,create_tasks,get_time]

APPROVAL_REQUIRED_TOOL_NAMES = {
    "google_calendar_create_calendar_event",
}

llm=ChatOpenAI(
    model=Settings.LLM_MODEL,
    api_key=Settings.LLM_API_KEY,
    base_url=Settings.LLM_BASE_URL,
    timeout=60,
    max_retries=2,
)

TITLE_SYSTEM_PROMPT = """
根据用户的首条消息生成一个简洁的会话标题。
要求：
1. 准确概括用户主题；
2. 中文标题控制在 6 到 20 个字，英文标题控制在 3 到 10 个单词；
3. 不要使用引号、句号或“标题：”前缀；
4. 只返回标题，不要解释。
""".strip()


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
        return "新对话"

    first_message_text = extract_content_text(first_user_message.content).strip()
    fallback_title = first_message_text[:20] or "新对话"

    try:
        response = llm.invoke(
            [
                SystemMessage(content=TITLE_SYSTEM_PROMPT),
                HumanMessage(content=first_message_text),
            ]
        )
    except Exception:
        logger.exception("Failed to generate conversation title")
        return fallback_title

    title = extract_content_text(response.content).strip()
    title = title.removeprefix("标题：").removeprefix("标题:")
    title = title.splitlines()[0].strip(" #\"'“”‘’。") if title else ""
    return title[:50] or fallback_title



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
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )

        update = {
            "messages": [response]
        }

        if not state.get("title"):
            update["title"] = generate_conversation_title(state)

        return update

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

    # 工具执行完成后，重新调用模型生成最终答案
    builder.add_edge(
        "tools",
        "agent",
    )

    return builder.compile(checkpointer=checkpointer)
