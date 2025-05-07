from typing import List, Union, Tuple, AsyncIterator, Callable
from core.llms.base import AsyncBaseChatCOTModel
from utils.log import logger
from openai import AsyncOpenAI
from core.schema import Message, ToolChoice, ToolCall, Function
from core.config import config

class OpenAICoT(AsyncBaseChatCOTModel):
    """支持链式思考的OpenAI模型实现，集成了原OpenAi类的功能"""
    
    def __init__(self, api_base: str,api_key: str,model: str, support_fn_call: bool | None = None,max_length: int = 8192):
        super().__init__(model, support_fn_call, max_length=max_length)
        logger.info(f'Initializing OpenAI CoT client | Model: {self.model} | URL: {api_base} ')
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def _process_stream_response(self, response) -> AsyncIterator[Tuple[str, str, List[ToolCall]]]:
        """处理流式响应，生成（思考片段，回答片段）元组"""
        buffer_reasoning = ""
        buffer_content = ""
        buffer_limit = 10  # 缓冲字符数
        tool_info = []
        
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            # 处理思考内容
            reasoning = getattr(delta, 'reasoning_content', "")
            if reasoning:
                buffer_reasoning += reasoning
                if len(buffer_reasoning) >= buffer_limit:
                    yield (buffer_reasoning, "", None)
                    buffer_reasoning = ""
            
            # 处理正式回答
            content = getattr(delta, 'content', "")
            if content:
                # 确保在处理content前先输出所有reasoning
                if buffer_reasoning:
                    yield (buffer_reasoning, "", None)
                    buffer_reasoning = ""
                buffer_content += content
                if len(buffer_content) >= buffer_limit:
                    yield ("", buffer_content, None)
                    buffer_content = ""

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    index = tool_call.index
                    while len(tool_info) <= index:
                        tool_info.append({})
                    if tool_call.id:
                        tool_info[index]["id"] = tool_info[index].get("id", "") + tool_call.id
                    if tool_call.function and tool_call.function.name:
                        tool_info[index]["name"] = tool_info[index].get("name", "") + tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        tool_info[index]["arguments"] = tool_info[index].get("arguments", "") + tool_call.function.arguments

        # 处理剩余缓冲
        if buffer_content:
            yield ("", buffer_content, None)
        if tool_info:
            tools = []
            for tool in tool_info:
                tools.append(ToolCall(
                    id=tool["id"],
                    function=Function(
                        name=tool["name"],
                        arguments=tool["arguments"]
                    )
                ))
            yield ("", "", tools)

    async def _chat_stream(self, 
                    messages: List[ dict], 
                    stop: List[str] | None = None,
                    tools: List[dict] | None = None, 
                    tool_choice: ToolChoice = ToolChoice.AUTO,
                    **kwargs) -> AsyncIterator[Tuple[str, str, list]]:
        logger.info(f'Calling OpenAI CoT API | Model: {self.model} | Stream: True | Messages: {messages}')
        
        temperature = kwargs.pop('temperature', config.model.temperature)
        max_tokens = kwargs.pop('max_tokens', config.model.max_tokens)
        timeout = kwargs.pop('timeout', config.model.timeout)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            **kwargs
        )

        return self._process_stream_response(response)

    async def _chat_no_stream(self, 
                        messages: List[dict], 
                        stop: List[str] | None = None,
                        tools: List[dict] | None = None, 
                        tool_choice: ToolChoice = ToolChoice.AUTO,
                        **kwargs) -> Tuple[str, str, list]:
        logger.info(f'Calling OpenAI CoT API | Model: {self.model} | Stream: False | Messages: {messages}')
        
        temperature = kwargs.pop('temperature', config.model.temperature)
        max_tokens = kwargs.pop('max_tokens', config.model.max_tokens)
        timeout = kwargs.pop('timeout', config.model.timeout)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            **kwargs
        )
            
        message = response.choices[0].message
        return (
            getattr(message, 'reasoning_content', ''),
            message.content,
            getattr(message, 'tool_calls', [])
        )