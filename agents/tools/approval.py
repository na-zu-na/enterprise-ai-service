from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt


def require_approval(tool: BaseTool, *, action: str | None = None) -> BaseTool:
    """Wrap a tool so its side effect runs only after explicit user approval."""

    async def call_after_approval(**arguments: Any) -> Any:
        decision = interrupt(
            {
                "action": action or tool.name,
                "messages": f"执行 {tool.name} 需要用户同意",
                "payload": arguments,
            }
        )

        if (
            not isinstance(decision, dict)
            or decision.get("approved") is not True
        ):
            return {
                "status": "cancelled",
                "message": "用户拒绝执行操作",
                "action": action or tool.name,
                "payload": arguments,
            }

        # The MCP call must stay after interrupt: this node is restarted on resume.
        return await tool.ainvoke(arguments)

    return StructuredTool.from_function(
        coroutine=call_after_approval,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        infer_schema=False,
        # The wrapper can also return a cancellation dict, so it uses the
        # regular content format even if the underlying MCP tool has an artifact.
        response_format="content",
        metadata=tool.metadata,
        tags=tool.tags,
    )
