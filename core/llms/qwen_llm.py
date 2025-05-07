from typing import AsyncIterator, List, Tuple
from .openai_llm import OpenAICoT
from core.schema import Message, ToolChoice
from utils.log import logger

class QwenCoT(OpenAICoT):
    def __init__(self, api_base: str, api_key: str, model: str, enable_thinking: bool = False, support_fn_call: bool | None = None, max_length: int = 8192):
        super().__init__(api_base, api_key, model, support_fn_call, max_length)
        self.enable_thinking = enable_thinking

    async def _chat_no_stream(self, messages: List[dict], stop: List[str] | None = None, **kwargs) -> Tuple[str, str]:
        all_thinking = ""
        all_result = ""
        generater = self._chat_stream(messages, stop, extra_body={"enable_thinking": False}, **kwargs)
        async for thinking, result in generater:
            all_thinking += thinking
            all_result += result
        return all_thinking, all_result

    async def _chat_stream(self, messages: List[dict], stop: List[str] | None = None, **kwargs) -> AsyncIterator[Tuple[str, str]]:
        extra_body = kwargs.pop("extra_body", {"enable_thinking": self.enable_thinking})
        return await super()._chat_stream(messages, stop, extra_body=extra_body, **kwargs)

    async def chat_with_tools(self, messages: List[dict], tools: List[dict] | None = None, tool_choice: ToolChoice = ToolChoice.AUTO, **kwargs) -> Message:
        return await super().chat_with_tools(messages, tools, tool_choice, extra_body={"enable_thinking": False}, stream=True, **kwargs)
    
    async def chat_with_tools_with_thinking(self, messages: List[dict], tools: List[dict] | None = None, tool_choice: ToolChoice = ToolChoice.AUTO, **kwargs) -> Message:
        return await self._chat_stream(messages, tools = tools, tool_choice = tool_choice, extra_body={"enable_thinking": True}, **kwargs)
