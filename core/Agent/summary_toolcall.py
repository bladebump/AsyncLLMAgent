from typing import List, Optional, Union, Any, Tuple, AsyncIterator
import asyncio
from core.agent.toolcall import ToolCallAgent
from core.tools import ToolCollection
from core.schema import AgentState, Message, AgentDone, ToolChoice
from utils.log import logger
from core.llms import AsyncBaseChatCOTModel
from core.mem import AsyncMemory


SUMMARY_SYSTEM_PROMPT = """你是一个可以执行工具调用的代理，请根据用户的需求选择合适的工具，并使用工具调用执行任务。

请遵循以下原则：
1. 首先检查是否有适合完成用户请求的工具
2. 如果有合适的工具，使用工具获取信息
3. 如果没有合适的工具，请直接使用`terminate`工具结束对话，状态设置为status="success"
4. 当使用有效工具收集信息并完成任务后，使用`terminate`工具结束对话，状态为status="success"
5. 只有在尝试使用工具但失败时，才使用status="failure"结束对话

不要尝试在对话中直接回答问题，让总结步骤处理最终的回答。"""

SUMMARY_NEXT_STEP_PROMPT = """在继续之前，请评估当前情况：

1. 如果你已经收集到足够的信息，使用`terminate`工具结束对话（status="success"）
2. 如果没有合适的工具，直接使用`terminate`工具结束对话（status="success"）
3. 只有在尝试使用工具但失败时，才使用`terminate`工具结束对话（status="failure"）

总结环节将负责提供最终回答，你不需要在对话中直接回答问题。"""

SUMMARIZE_PROMPT = """请基于以上对话历史为用户提供完整回答：

1. 如果用户提出了具体问题：
   - 如果没有使用相关工具，请直接利用你自身的知识回答问题，给出全面、准确的回应
   - 如果使用了相关工具，请基于工具的结果提供回答
   
2. 如果用户没有提问：请提供简洁明了的总结，包括：
   - 已完成的任务
   - 获取的关键信息
   - 得出的主要结论

你的回答应当：
- 专注于核心结果和重要发现，省略中间过程细节
- 提供足够完整的信息，使没有查看对话历史的用户也能完全理解结果
- 确保回答是有帮助的，即使之前没有使用专门的工具

用户问题：
{request}
"""


class SummaryToolCallAgent(ToolCallAgent):
    """使用terminate工具结束对话，然后通过LLM生成总结的代理类"""

    def __init__(
        self,
        name: str = "summary_toolcall",
        llm: AsyncBaseChatCOTModel = None,
        memory: AsyncMemory = None,
        description: str = "一个可以执行工具调用，执行完成后提供总结的代理。",
        system_prompt: str = SUMMARY_SYSTEM_PROMPT,
        next_step_prompt: str = SUMMARY_NEXT_STEP_PROMPT,
        state: AgentState = AgentState.IDLE,
        available_tools: Optional[ToolCollection] = None,
        tool_choices: str = ToolChoice.AUTO,
        special_tool_names: Optional[List[str]] = None,
        max_steps: int = 30,
        max_observe: Optional[Union[int, bool]] = None,
        **kwargs
    ):      
        super().__init__(
            name=name,
            llm=llm,
            memory=memory,
            description=description,
            system_prompt=system_prompt,
            next_step_prompt=next_step_prompt,
            state=state,
            available_tools=available_tools,
            special_tool_names=special_tool_names,
            tool_choices=tool_choices,
            max_steps=max_steps,
            max_observe=max_observe,
            **kwargs
        )
    
    async def _generate_summary(self, stream: bool = False) -> Union[Tuple[str, str], AsyncIterator[Tuple[str, str]]]:
        """根据对话历史生成总结"""
        try:
            # 创建总结提示
            summary_prompt = Message.user_message(SUMMARIZE_PROMPT.format(request=self.request))
            
            # 获取完整对话历史
            history = self.memory.Messages.copy()
            
            # 添加总结提示
            history.append(summary_prompt)
            
            # 使用LLM生成总结
            logger.info("生成对话总结...")
            response = await self.llm.chat(messages=history, stream=stream)
                
            return response
        except Exception as e:
            logger.error(f"生成总结时发生错误: {e}")
            return f"无法生成总结: {str(e)}"
    
    async def run(self, request: Optional[str] = None) -> str:
        """重写run方法以返回summary结果"""
        _ = await super().run(request)
        result = await self._generate_summary()
        return result
    
    async def _run_and_fill_queue(self, queue: asyncio.Queue) -> None:
        """内部方法，执行步骤并将结果放入队列。"""
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                # Check for stuck state
                if await self.is_stuck():
                    await self.handle_stuck_state()

                await queue.put(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                await queue.put(f"终止: 达到最大步骤 ({self.max_steps})")
            
            summary = await self._generate_summary(stream=True)
            await queue.put(summary)
            # 标记队列已完成
            await queue.put(AgentDone(reason="执行完成"))