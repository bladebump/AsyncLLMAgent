from typing import List, Union, Tuple, AsyncIterator, Callable
from core.llms.base import AsyncBaseChatCOTModel
from utils.log import logger
from utils.retry import retry
from openai import AsyncOpenAI
from core.openai_types import Message, MessageToolParam
from core.util import function_to_json
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
                    messages: List[Message], 
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
                       messages: List[Message], 
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

    async def support_function_calling(self):
        """检查模型是否支持函数调用"""
        if self._support_fn_call is None:
            return await super().support_function_calling()
        else:
            return self._support_fn_call

    @retry(max_retries=3, delay_seconds=0.5)
    async def chat(self, 
            prompt: str | None = None,
            messages: List[Message] | None = None,
            stop: List[str] | None = None,
            stream: bool = False,
            **kwargs) -> Union[Tuple[str, str], AsyncIterator[Tuple[str, str]]]:
        """统一的对话接口，支持原始提示词和消息格式"""
            
        # 处理消息格式
        if not messages and prompt and isinstance(prompt, str):
            messages = [Message(role='user', content=prompt)]
            
        # 强制使用消息格式
        assert messages and len(messages) > 0, "Messages cannot be empty"
        
        if isinstance(messages[0], Message):
            messages = [item.model_dump() for item in messages]
        # 执行父类逻辑
        return await super().chat(
            messages=messages,
            stop=stop,
            stream=stream,
            **kwargs
        )

    async def chat_with_functions(self, messages: List[Message], functions: List[MessageToolParam], **kwargs) -> Message:
        """支持MCP工具调用的对话接口"""
        if not isinstance(messages[0], dict):
            messages = [item.model_dump() for item in messages]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=functions,
            tool_choice='auto',
            **kwargs
        )
        return response.choices[0].message
