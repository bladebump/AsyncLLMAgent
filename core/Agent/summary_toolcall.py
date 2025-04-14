from typing import List
from pydantic import Field
from core.agent.toolcall import ToolCallAgent
from utils.log import logger
from core.schema import Message, ToolChoice
from core.tools import Terminate, CreateChatCompletion, Summarize, ToolCollection

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

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate(), Summarize()
    )
    
    # 添加Summarize到特殊工具列表
    special_tool_names: List[str] = Field(
        default_factory=lambda: [Terminate().name, Summarize().name]
    )

    def _should_finish_execution(self, name: str, **kwargs) -> bool:
        """确定是否应该完成工具执行
        
        当调用Terminate或Summarize工具时，执行应该结束
        """
        # 检查是否为终止或总结工具
        terminate_name = Terminate().name.lower()
        summarize_name = Summarize().name.lower()
        return name.lower() in [terminate_name, summarize_name] 