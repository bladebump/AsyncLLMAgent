from typing import List, Optional, Union, Any
import asyncio
from pydantic import Field
from core.agent.toolcall import ToolCallAgent
from core.tools import Summarize, ToolCollection
from core.schema import AgentState, Message, AgentDone
from utils.log import logger

SUMMARY_SYSTEM_PROMPT = """你是一个可以执行工具调用的代理，请根据用户的需求选择合适的工具，并使用工具调用执行任务。
你可以反复使用工具调用直到任务完成。
在完成所有必要的工具调用后，请使用summarize工具提供一个清晰、简洁的最终答案。"""

SUMMARY_NEXT_STEP_PROMPT = """如果你想停止交互，请使用`terminate`工具/函数调用。
如果你已经收集到足够的信息，可以使用`summarize`工具提供一个完整的回答。"""


class SummaryToolCallAgent(ToolCallAgent):
    """使用summarize工具提供最终答案的代理类"""

    name: str = "summary_toolcall"
    description: str = "一个可以执行工具调用并使用summarize工具提供最终答案的代理。"

    system_prompt: str = SUMMARY_SYSTEM_PROMPT
    next_step_prompt: str = SUMMARY_NEXT_STEP_PROMPT

    available_tools: ToolCollection = Field(default_factory=lambda: ToolCollection(
        Summarize()
    ))

    special_tool_names: List[str] = Field(default_factory=lambda: [Summarize().name])
    
    def __init__(self, **kwargs):
        """初始化代理并设置实例属性"""
        super().__init__(**kwargs)
        # 存储最终总结结果，作为实例属性而非类属性
        self.summary_result: Optional[str] = None
    
    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """处理特殊工具执行和状态变化，捕获summarize工具的结果"""
        if self._is_special_tool(name):
            self.summary_result = str(result)
            logger.info(f"捕获到summarize工具结果: {self.summary_result[:100]}...")
        
        # 调用父类方法处理特殊工具
        await super()._handle_special_tool(name=name, result=result, **kwargs)
    
    async def run(self, request: Optional[str] = None) -> str:
        """重写run方法以返回summary结果"""
        # 重置summary_result
        self.summary_result = None
        
        # 调用父类的run方法执行代理流程
        _ = await super().run(request)
        
        # 如果有summary结果，则返回它；否则返回最后一条消息
        if self.summary_result:
            return self.summary_result
        
        return "未能获取总结结果"
    
    async def run_stream(self, request: Optional[str] = None) -> asyncio.Queue:
        """重写run_stream方法以支持流式输出，最后返回summary结果"""
        # 重置summary_result
        self.summary_result = None
        # 获取父类的stream队列
        result_queue = await super().run_stream(request)
        return result_queue
    
    async def get_summary_result(self) -> Optional[str]:
        """获取总结结果"""
        return self.summary_result
