from typing import List, Union, Tuple, AsyncIterator, Callable
from core.llms.base import AsyncBaseChatCOTModel
from utils.log import logger
from openai import AsyncOpenAI
from core.schema import Message
from config import LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT

class OpenAICoT(AsyncBaseChatCOTModel):
    """支持链式思考的OpenAI模型实现，集成了原OpenAi类的功能"""
    
    def __init__(self, api_base: str,api_key: str,model: str, support_fn_call: bool | None = None,max_length: int = 8192):
        super().__init__(model, support_fn_call, max_length=max_length)
        logger.info(f'Initializing OpenAI CoT client | Model: {self.model} | URL: {api_base} ')
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def _process_stream_response(self, response) -> AsyncIterator[Tuple[str, str]]:
        """处理流式响应，生成（思考片段，回答片段）元组"""
        buffer_reasoning = ""
        buffer_content = ""
        buffer_limit = 10  # 缓冲字符数
        
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            # 处理思考内容
            reasoning = getattr(delta, 'reasoning_content', "") or ""
            if reasoning:
                buffer_reasoning += reasoning
                if len(buffer_reasoning) >= buffer_limit:
                    yield (buffer_reasoning, "")
                    buffer_reasoning = ""
            
            # 处理正式回答
            content = getattr(delta, 'content', "") or ""
            if content:
                buffer_content += content
                if len(buffer_content) >= buffer_limit:
                    yield ("", buffer_content)
                    buffer_content = ""
        
        # 处理剩余缓冲
        if buffer_reasoning:
            yield (buffer_reasoning, "")
        if buffer_content:
            yield ("", buffer_content)

    async def _chat_stream(self, 
                    messages: List[ dict], 
                    stop: List[str] | None = None,
                    **kwargs) -> AsyncIterator[Tuple[str, str]]:
        logger.info(f'Calling OpenAI CoT API | Model: {self.model} | Stream: True | Messages: {messages}')
        
        temperature = kwargs.pop('temperature', LLM_TEMPERATURE)
        max_tokens = kwargs.pop('max_tokens', LLM_MAX_TOKENS)
        timeout = kwargs.pop('timeout', LLM_TIMEOUT)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop,
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
                       **kwargs) -> Tuple[str, str]:
        logger.info(f'Calling OpenAI CoT API | Model: {self.model} | Stream: False | Messages: {messages}')
        
        temperature = kwargs.pop('temperature', LLM_TEMPERATURE)
        max_tokens = kwargs.pop('max_tokens', LLM_MAX_TOKENS)
        timeout = kwargs.pop('timeout', LLM_TIMEOUT)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            **kwargs
        )
            
        message = response.choices[0].message
        return (
            getattr(message, 'reasoning_content', '') or '',  # 兼容无推理内容的情况
            message.content
        )

    async def chat_with_tools(self, messages: List[Union[Message, dict]], tools: List[dict] | None = None, **kwargs) -> Message:
        """支持MCP工具调用的对话接口"""
        if not isinstance(messages[0], dict):
            messages = self.format_messages(messages)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice='auto',
            **kwargs
        )
        return response.choices[0].message
