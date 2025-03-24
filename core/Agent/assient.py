from core.Agent.base import AsyncAgent
from core.openai_types import Message
from typing import Iterator, Union, List, Tuple

class AsyncAssistant(AsyncAgent):
    """
    一个异步的助手，可以执行任务
    """
    
    async def _run(self, prompt: str, messages: List[Message] | None = None, **kwargs) -> Union[Tuple[str, str], Iterator[Tuple[str, str]]]:
        if messages is None:    
            messages = [
                Message(role="system", content=self.instruction),
                Message(role="user", content=prompt)
            ]
        else:
            messages.append(Message(role="user", content=prompt))
        return await self._call_llm(messages=messages, **kwargs)


