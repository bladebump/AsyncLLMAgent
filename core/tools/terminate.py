from core.tools.base import BaseTool


_TERMINATE_DESCRIPTION = """当请求被满足或助手无法进一步执行任务时，终止交互。
当你完成所有任务后，调用此工具结束工作。"""


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "交互的完成状态。",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    async def execute(self, status: str) -> str:
        """结束当前执行"""
        return f"交互已完成，状态为: {status}"