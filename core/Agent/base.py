from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Union ,Tuple
from core.llms.base import AsyncBaseLLMModel
from core.mem.base import AsyncMemory
from core.openai_types import Message, Function, MessageToolParam

class AsyncAgent(ABC):

    def __init__(self, tool_list: List[MessageToolParam] | None = None, llm:AsyncBaseLLMModel = None, memory: AsyncMemory = None,
                name: str | None = None, instruction: Union[str, dict] = None, stream: bool = True, **kwargs):
        """
        初始化一个异步Agent

        Args:
            tool_list: 一个工具列表
            llm: 这个Agent的LLM配置
            memory: 这个Agent的记忆
            name: 这个Agent的名称
            instruction: 这个Agent的系统指令
            stream: 是否流式输出
            kwargs: 其他潜在的参数
        """
        self.llm = llm
        self.memory = memory
        self.stream = stream
        self.tool_list = tool_list
        self.name = name
        self.instruction = instruction
        self.function_map = {}

    async def run(self, *args, **kwargs) -> Union[Tuple[str, str], Iterator[Tuple[str, str]]]:
        result = await self._run(*args, **kwargs)
        return result

    @abstractmethod
    async def _run(self, *args, **kwargs) -> Union[Tuple[str, str], Iterator[Tuple[str, str]]]:
        raise NotImplementedError

    async def _call_llm(self, prompt: str | None = None, messages: List[Message] | None = None, stop: List[str] | None = None, **kwargs) -> Union[Tuple[str, str], Iterator[Tuple[str, str]]]:
        return await self.llm.chat(prompt=prompt, messages=messages, stop=stop, stream=self.stream, **kwargs)

    async def _call_llm_with_tools(self, messages: List[Message], tools: List[MessageToolParam], **kwargs) -> Message:
        return await self.llm.chat_with_tools(messages=messages, tools=tools, **kwargs)

    async def _call_tool(self, tool_list: list[Function], **kwargs) -> list:
        raise NotImplementedError

    async def _detect_tool(self, message: Message) -> Tuple[bool, List[Function], str]:
        """
        内置工具调用检测

        Args:
            message: 一个消息
                (1) 当message是Message时：通过函数调用格式确定是否调用工具。
                (2) 当message是str时：需要从字符串中解析工具，并在此处实现一个自定义的_detect_tool函数。

        Returns:
            - bool: 是否需要调用工具
            - list[Function]: 工具列表
            - str: 工具调用之外的文本回复
        """

        func_calls = []
        if message.tool_calls:
            for item in message.tool_calls:
                func_call = item.function
                func_calls.append(func_call)
        text = message.content or ''
        return (len(func_calls) > 0), func_calls, text
