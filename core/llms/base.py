from abc import ABC, abstractmethod
from utils.retry import retry
from typing import Iterator, List, Union, Tuple, Callable, Literal, AsyncIterator
from utils.log import logger
from core.openai_types import Message, MessageToolParam

class FnCallNotImplError(NotImplementedError):
    pass

class TextCompleteNotImplError(NotImplementedError):
    pass

def get_current_weather(location: str, unit: Literal['celsius', 'fahrenheit'] = 'celsius'):
    """Get the current weather in a given location."""
    return f"The current weather in {location} is 20 degrees {unit}."


class AsyncBaseLLMModel(ABC):
    """LLM基础模型，包含通用功能"""
    
    def __init__(self,
                 model: str,
                 support_fn_call: bool | None = None,
                 max_length: int = 8192):
        self._support_fn_call = support_fn_call
        self.model = model
        self.max_length = max_length

    async def support_function_calling(self) -> bool:
        if self._support_fn_call is None:
            messages = [{'role': 'user', 'content': 'What is the weather like in Boston?'}]
            self._support_fn_call = False
            try:
                response = await self.chat_with_functions(messages=messages, functions=[get_current_weather])
                if response.function_call or response.tool_calls:
                    self._support_fn_call = True
            except Exception as e:
                logger.error(f'Function calling check failed: {e}')
        return self._support_fn_call

    def check_max_length(self, messages: List[Message]) -> bool:
        total_length = sum(len(msg.content) for msg in messages)
        return total_length <= self.max_length

    def get_max_length(self) -> int:
        return self.max_length
    
    @abstractmethod
    async def chat(
        self,
        prompt: str | None = None,
        messages: List[Message] | None = None,
        stop: List[str] | None = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[str, str], AsyncIterator[Tuple[str, str]]]:
        raise NotImplementedError
    
    @abstractmethod
    async def chat_with_functions(
        self,
        messages: List[Message],
        functions: List[MessageToolParam],
        **kwargs
    ) -> Message:
        raise NotImplementedError

class AsyncBaseChatCOTModel(AsyncBaseLLMModel):
    """链式思考（CoT）模型基类，返回（思考过程，最终响应）"""

    @abstractmethod
    async def _chat_stream(
        self,
        messages: List[Message],
        stop: List[str] | None = None,
        **kwargs
    ) -> AsyncIterator[Tuple[str, str]]:
        """流式返回（思考token，响应token）的生成器"""
        raise NotImplementedError

    @abstractmethod
    async def _chat_no_stream(
        self,
        messages: List[Message],
        stop: List[str] | None = None,
        **kwargs
    ) -> Tuple[str, str]:
        """非流式返回（完整思考，完整响应）的元组"""
        raise NotImplementedError

    @retry(max_retries=3, delay_seconds=0.5)
    async def chat(
        self,
        prompt: str | None = None,
        messages: List[Message] | None = None,
        stop: List[str] | None = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[str, str], AsyncIterator[Tuple[str, str]]]:
        # 处理消息格式
        if not messages and prompt and isinstance(prompt, str):
            messages = [Message(role='user', content=prompt)]
            
        # 强制使用消息格式
        assert messages and len(messages) > 0, "Messages cannot be empty"
        
        if isinstance(messages[0], Message):
            messages = [item.model_dump() for item in messages]

        if stream:
            return await self._chat_stream(messages, stop=stop, **kwargs)
        else:
            return await self._chat_no_stream(messages, stop=stop, **kwargs)