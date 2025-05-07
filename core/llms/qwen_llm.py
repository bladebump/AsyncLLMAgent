from typing import AsyncIterator, List, Tuple
from .openai_llm import OpenAICoT
from core.schema import Message, ToolChoice
from utils.log import logger

class QwenCoT(OpenAICoT):
    def __init__(self, api_base: str, api_key: str, model: str, enable_thinking: bool = False, support_fn_call: bool | None = None, max_length: int = 8192):
        super().__init__(api_base, api_key, model, support_fn_call, max_length)
        self.enable_thinking = enable_thinking

    async def _chat_no_stream(self, messages: List[dict], stop: List[str] | None = None, tools: List[dict] | None = None, tool_choice: ToolChoice = ToolChoice.AUTO, **kwargs) -> Tuple[str, str, list]:
        all_thinking = ""
        all_result = ""
        generater = await self._chat_stream(messages, stop, tools, tool_choice, extra_body={"enable_thinking": self.enable_thinking}, **kwargs)
        async for thinking, result, tool_calls in generater:
            all_thinking += thinking
            all_result += result
        return all_thinking, all_result, tool_calls

    async def _chat_stream(self, messages: List[dict], stop: List[str] | None = None, tools: List[dict] | None = None, tool_choice: ToolChoice = ToolChoice.AUTO, **kwargs) -> AsyncIterator[Tuple[str, str, list]]:
        extra_body = kwargs.pop("extra_body", {"enable_thinking": self.enable_thinking})
        return await super()._chat_stream(messages, stop, tools, tool_choice, extra_body=extra_body, **kwargs)
