from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

def create_google_calendar_mcp_client()->MultiServerMCPClient:
    client = MultiServerMCPClient(
        {
            "google_calendar": {
                "transport": "stdio",
                "command":  r"C:\Users\25537\桌面\实用工具\Project\code\calendar-mcp-server\.venv\Scripts\python.exe",
                "args": [
                    r"C:\Users\25537\桌面\实用工具\Project\code\calendar-mcp-server\server.py",
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