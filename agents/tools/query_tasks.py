from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from agents.state import AgentRuntimeContext
from clients.spring_client import get_spring_client, SpringClientError

client=get_spring_client()

@tool
def query_tasks(
        runtime: ToolRuntime[AgentRuntimeContext],
)->list[str]:
    """查询当前登录用户可以访问的任务列表。"""
    access_token = runtime.context["access_token"]
    response=client.get("/api/tasks",access_token=access_token)
    body=response.json()

    if(body.get("code")!=200):
        raise SpringClientError(
            body.get("message", "Spring 请求失败")
        )

    data=body.get("data")
    return data
