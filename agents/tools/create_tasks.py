from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from agents.schemas.tasks import tasks
from agents.state import AgentRuntimeContext
from clients.spring_client import get_spring_client, SpringClientError

client=get_spring_client()

@tool
def create_tasks(
        tasks:tasks,
        runtime: ToolRuntime[AgentRuntimeContext],
)-> dict:
    """根据用户提供的标题、描述和截止时间创建任务。"""
    access_token = runtime.context["access_token"]
    response=client.post("/api/tasks",
                         access_token=access_token,
                         json=tasks.model_dump(
                             mode="json",
                             by_alias=True,
                         ))
    body=response.json()
    if(body.get("code")!=200):
        raise SpringClientError(
            body.get("message", "Spring 请求失败")
        )

    data=body.get("data")

    if not isinstance(data, dict):
        raise SpringClientError(
            "Spring 创建任务响应缺少有效的 data"
        )

    return data
