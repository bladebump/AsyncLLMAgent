from core.Agent.base import AsyncAgent
from core.llms.base import AsyncBaseLLMModel
from core.mem.base import AsyncMemory
from core.openai_types import Message, MessageToolParam, MessageToolCall
from typing import Iterator, Union, List, Tuple
from mcp import ClientSession

class AsyncAssistantTool(AsyncAgent):
    """
    一个异步的助手，可以执行任务, 并使用工具
    """

    def __init__(self, 
                 tool_list: List[MessageToolParam] | None = None, 
                 llm: AsyncBaseLLMModel = None, 
                 memory: AsyncMemory = None, 
                 name: str | None = None, 
                 instruction: str | dict = None, 
                 stream: bool = True, 
                 mcp_sessions: ClientSession | None = None,
                 **kwargs):
        self.mcp_sessions = mcp_sessions
        super().__init__(tool_list, llm, memory, name, instruction, stream, **kwargs)
    
    async def _run(self, prompt: str, messages: List[Message] | None = None, **kwargs) -> Union[Tuple[str, str], Iterator[Tuple[str, str]]]:
        if messages is None:    
            messages = [
                Message(role="system", content=self.instruction),
                Message(role="user", content=prompt)
            ]
        else:
            messages.append(Message(role="user", content=prompt))
        max_turn = kwargs.get("max_turn", 5)
        turn = 0
        while turn < max_turn:
            response = await self._call_llm_with_tools(messages=messages, tools=self.tool_list)
            messages.append(response)
            if response.tool_calls:
                tool_calls = response.tool_calls
                result = await self._call_tool(tool_calls)
                messages.extend(result)
            else:
                break
            turn += 1
        return messages

    
    async def _call_tool(self, tool_list: list[MessageToolCall], **kwargs) -> list:
        messages = []
        for tool in tool_list:
            tool_name = tool.function.name
            tool_args = tool.function.arguments
            result = await self.mcp_sessions.call_tool(tool_name, tool_args)
            
            tool_message = {
                "role": "tool",
                "tool_call_id": tool.id,
                "content": "\n".join([item.model_dump_json() for item in result.content])
            }
            messages.append(tool_message)


