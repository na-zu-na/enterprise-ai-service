from langchain_mcp_adapters.client import MultiServerMCPClient

from core.config import Settings


def create_google_calendar_mcp_client()->MultiServerMCPClient:
    if not Settings.GOOGLE_CALENDAR_MCP_PYTHON:
        raise RuntimeError("缺少 GOOGLE_CALENDAR_MCP_PYTHON 环境变量")
    if not Settings.GOOGLE_CALENDAR_MCP_SERVER:
        raise RuntimeError("缺少 GOOGLE_CALENDAR_MCP_SERVER 环境变量")

    client = MultiServerMCPClient(
        {
            "google_calendar": {
                "transport": "stdio",
                "command": Settings.GOOGLE_CALENDAR_MCP_PYTHON,
                "args": [
                    Settings.GOOGLE_CALENDAR_MCP_SERVER,
                ]
            }
        },
        tool_name_prefix=True,
    )

    return client

async def load_google_calendar_tools(client):
    tools = await client.get_tools()

    if not tools:
        raise RuntimeError(
            "Google Calendar MCP 没有暴露任何工具"
        )

    return tools
