from core.tools.base import BaseTool
from core.llms import AsyncBaseChatCOTModel
from pydantic import Field
from core.schema import Message

_SUMMARIZE_DESCRIPTION = """总结之前所有工具调用的结果，提供一个最终的答案给用户。
在你已经收集了足够的信息但还没有准备终止对话时，使用此工具来生成一个总结性回答。"""


class Summarize(BaseTool):
    name: str = "summarize"
    description: str = _SUMMARIZE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "总结的内容，应该包含对用户问题的完整回答。",
            }
        },
        "required": ["message"],
    }
    llm: AsyncBaseChatCOTModel = Field(...)

    async def execute(self, message: list[Message]) -> str:
        """生成总结回答"""
        _, response = await self.llm.chat(message, stream=False)
        return response
