from abc import ABC, abstractmethod
from utils.retry import retry
from typing import List, Union, Tuple, Literal, AsyncIterator
from utils.log import logger
from core.schema import Message, ROLE_VALUES, TOOL_CHOICE_TYPE, ToolChoice

class FnCallNotImplError(NotImplementedError):
    pass

class TextCompleteNotImplError(NotImplementedError):
    pass

class AsyncBaseLLMModel(ABC):
    """LLM基础模型，包含通用功能"""
    
    def __init__(self,
                 model: str,
                 support_fn_call: bool | None = None,
                 max_length: int = 8192):
        self._support_fn_call = support_fn_call
        self.model = model
        self.max_length = max_length

    def check_max_length(self, messages: List[Message]) -> bool:
        total_length = sum(len(msg.content) for msg in messages)
        return total_length <= self.max_length

    def get_max_length(self) -> int:
        return self.max_length
    
    @abstractmethod
    async def chat(
        self,
        prompt: str | None = None,
        messages: List[Union[Message, dict]] | None = None,
        stop: List[str] | None = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[str, str], AsyncIterator[Tuple[str, str]]]:
        raise NotImplementedError
    
    @abstractmethod
    async def chat_with_tools(
        self,
        messages: List[Union[Message, dict]],
        tools: List[dict] | None = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO, # type: ignore
        **kwargs
    ) -> Message:
        raise NotImplementedError
    
    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        formatted_messages = []
        for message in messages:
            if isinstance(message, Message):
                message = message.to_dict()
            if isinstance(message, dict):
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                if supports_images and message.get("base64_image"):
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )
                    del message["base64_image"]
                elif not supports_images and message.get("base64_image"):
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages


class AsyncBaseChatCOTModel(AsyncBaseLLMModel):
    """链式思考（CoT）模型基类，返回（思考过程，最终响应）"""

    @abstractmethod
    async def _chat_stream(
        self,
        messages: List[dict],
        stop: List[str] | None = None,
        **kwargs
    ) -> AsyncIterator[Tuple[str, str]]:
        """流式返回（思考token，响应token）的生成器"""
        raise NotImplementedError

    @abstractmethod
    async def _chat_no_stream(
        self,
        messages: List[dict],
        stop: List[str] | None = None,
        **kwargs
    ) -> Tuple[str, str]:
        """非流式返回（完整思考，完整响应）的元组"""
        raise NotImplementedError

    @retry(max_retries=3, delay_seconds=0.5)
    async def chat(
        self,
        prompt: str | None = None,
        messages: List[Union[Message, dict]] | None = None,
        stop: List[str] | None = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[str, str], AsyncIterator[Tuple[str, str]]]:
        # 处理消息格式
        if not messages and prompt and isinstance(prompt, str):
            messages = [Message.user_message(prompt)]
            
        # 强制使用消息格式
        assert messages and len(messages) > 0, "Messages cannot be empty"
        
        if isinstance(messages[0], Message):
            messages = self.format_messages(messages)

        if stream:
            return await self._chat_stream(messages, stop=stop, **kwargs)
        else:
            return await self._chat_no_stream(messages, stop=stop, **kwargs)