from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt

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
    task_payload=tasks.model_dump(
        mode="json",
        by_alias=True,
    )

    #发出请求，执行审批
    decision= interrupt({
        "action":"create_tasks",
        "messages":"创建任务需要用户同意",
        "payload":task_payload,
    })

    #默认拒绝：只有明确传入 approved=true才会执行
    if(
            not isinstance(decision, dict)
        or decision.get("approved") is not True
    ):
        return {
            "status": "cancelled",
            "message": "用户拒绝创建任务，任务未创建",
            "task": task_payload,
        }

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

    return {
        "status": "created",
        "data": data,
    }
