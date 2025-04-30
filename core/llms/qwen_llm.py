from typing import AsyncIterator, Coroutine, List, Tuple
from .openai_llm import OpenAICoT
from core.schema import Message, ToolChoice
from utils.log import logger

class QwenCoT(OpenAICoT):
    def __init__(self, api_base: str, api_key: str, model: str, enable_thinking: bool = False, support_fn_call: bool | None = None, max_length: int = 8192):
        super().__init__(api_base, api_key, model, support_fn_call, max_length)
        self.enable_thinking = enable_thinking

    async def _chat_no_stream(self, messages: List[dict], stop: List[str] | None = None, **kwargs) -> Tuple[str, str]:
        return await super()._chat_no_stream(messages, stop, extra_body={"enable_thinking": False},**kwargs)

    async def _chat_stream(self, messages: List[dict], stop: List[str] | None = None, **kwargs) -> AsyncIterator[Tuple[str, str]]:
        return await super()._chat_stream(messages, stop, extra_body={"enable_thinking": self.enable_thinking}, **kwargs)

    async def chat_with_tools(self, messages: List[dict], tools: List[dict] | None = None, tool_choice: ToolChoice = ToolChoice.AUTO, **kwargs) -> Message:
        return await super().chat_with_tools(messages, tools, tool_choice, extra_body={"enable_thinking": False}, **kwargs)

