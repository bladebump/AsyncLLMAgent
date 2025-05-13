import json
import time
import asyncio
from enum import Enum
from typing import Dict, List, Optional
from core.agent.base import BaseAgent
from core.flow.base import BaseFlow
from core.llms import AsyncBaseChatCOTModel
from utils.log import logger
from core.schema import AgentState, Message, ToolChoice, AgentResultStream, AgentDone, QueueEnd
from core.tools import PlanningTool


class PlanStepStatus(str, Enum):
    """枚举类定义计划步骤的可能状态"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """返回所有可能的步骤状态值列表"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """返回表示活动状态（未开始或进行中）的值列表"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """返回状态到其标记符号的映射"""
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


class PlanningFlow(BaseFlow):
    """一个管理计划和任务执行的流程"""

    def __init__(self, agents: BaseAgent | List[BaseAgent] | Dict[str, BaseAgent], 
                llm: AsyncBaseChatCOTModel, 
                tools: List | None = None, 
                primary_agent_key: str | None = None, 
                planning_tool: PlanningTool | None = None,
                executor_keys: List[str] | None = None,
                active_plan_id: str | None = None,
                current_step_index: int | None = None):
        super().__init__(agents, tools, primary_agent_key)
        self.llm = llm
        self.planning_tool = planning_tool or PlanningTool()
        self.executor_keys = executor_keys or list(self.agents.keys())
        self.active_plan_id = active_plan_id or f"plan_{int(time.time())}"
        self.current_step_index = current_step_index

    # 辅助方法：获取计划数据
    def _get_plan_data(self) -> Optional[dict]:
        """安全地获取当前活动计划的数据"""
        if not self.active_plan_id or self.active_plan_id not in self.planning_tool.plans:
            logger.error(f"计划 ID {self.active_plan_id} 未找到")
            return None
        return self.planning_tool.plans[self.active_plan_id]

    # 辅助方法：更新步骤状态
    async def _update_step_status(self, step_index: int, status: str) -> bool:
        """更新指定步骤的状态，失败时进行直接更新"""
        try:
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=step_index,
                step_status=status,
            )
            logger.info(f"在计划 {self.active_plan_id} 中标记步骤 {step_index} 为 {status}")
            return True
        except Exception as e:
            logger.warning(f"更新步骤状态时出错: {e}")
            # 直接在计划数据中更新状态
            plan_data = self._get_plan_data()
            if not plan_data:
                return False
                
            step_statuses = plan_data.get("step_statuses", [])
            
            # 确保step_statuses列表足够长
            while len(step_statuses) <= step_index:
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                
            # 更新状态
            step_statuses[step_index] = status
            plan_data["step_statuses"] = step_statuses
            return True

    def get_executor(self, agent_name: Optional[str] = None) -> BaseAgent:
        """
        获取适合当前步骤的执行代理。
        可以根据步骤类型/要求进行扩展。
        """
        # 如果提供了步骤类型并且与代理键匹配，则使用该代理
        if agent_name and agent_name in self.agents:
            return self.agents[agent_name]

        # 否则使用第一个可用的执行器或回退到主代理
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # 回退到主代理
        return self.primary_agent

    async def execute(self, input_text: str) -> asyncio.Queue:
        """执行计划流程，以流式方式返回结果"""
        result_queue = asyncio.Queue()
        
        try:
            if not self.primary_agent:
                raise ValueError("没有可用的主代理")

            # 启动一个独立的任务来执行流程并填充队列
            asyncio.create_task(self._execute_flow(input_text, result_queue))
            
            return result_queue
        except Exception as e:
            logger.error(f"PlanningFlow 错误: {str(e)}")
            error_queue = asyncio.Queue()
            error_stream = AgentResultStream(thinking="", content=f"执行失败: {str(e)}", tool_calls=[])
            await error_queue.put(error_stream)
            await error_queue.put(QueueEnd())
            await result_queue.put(error_queue)
            return result_queue

    async def _execute_flow(self, input_text: str, result_queue: asyncio.Queue):
        """具体执行流程的内部方法"""
            # 如果提供了输入，则创建初始计划
        if input_text:
            create_plan_queue = asyncio.Queue()
            await result_queue.put(create_plan_queue)
            await self._create_initial_plan(input_text, create_plan_queue)
                
            # 确认计划是否成功创建
            if self.active_plan_id not in self.planning_tool.plans:
                logger.error(
                    f"计划创建失败. 计划 ID {self.active_plan_id} 未在计划工具中找到."
                )
                await create_plan_queue.put(AgentResultStream(
                    thinking="", 
                    content=f"计划创建失败: {input_text}", 
                    tool_calls=[]
                ))
                await create_plan_queue.put(QueueEnd())
                await result_queue.put(AgentDone())
                return

        while True:
            # 获取当前要执行的步骤
            self.current_step_index, step_info = await self._get_current_step_info()

            # 如果没有任何步骤或计划完成，则退出
            if self.current_step_index is None:
                await self._finalize_plan(result_queue)
                break

            # 使用适当的代理执行当前步骤
            agent_name = step_info.get("agent_name") if step_info else None
            executor = self.get_executor(agent_name)
            await self._execute_step(executor, step_info, result_queue)

            # 检查代理是否想要终止
            if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                break

        await result_queue.put(AgentDone())

    async def _create_initial_plan(self, request: str, result_queue: asyncio.Queue):
        """使用流程的LLM和PlanningTool基于请求创建初始计划。"""
        logger.info(f"正在创建初始计划: {self.active_plan_id}")

        agent_dict = {agent.name: agent.description for agent in self.agents.values()}
        # Create a system message for plan creation
        system_message = Message.system_message(
            f"""你是一个计划助手。创建一个简洁、可操作的计划，具有清晰的步骤。 专注于关键里程碑，而不是详细的子步骤。 优化清晰度和效率。
每个steps请配合一个合适的agent。
当前有的agent有：{agent_dict}
""")
        # 创建一个包含请求的用户消息
        user_message = Message.user_message(
            f"创建一个合理的计划，具有清晰的步骤，以完成任务: {request}"
        )

        # 使用PlanningTool调用LLM (流式)
        gen = await self.llm.chat(
            messages=[system_message, user_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
            stream=True,
        )

        all_thinking = ""
        all_content = ""
        tool_calls = []
        
        # 处理流式响应
        async for thinking, content, calls in gen:
            all_thinking += thinking
            all_content += content
            if calls:
                tool_calls = calls
            
            # 将中间结果发送到队列
            await result_queue.put(AgentResultStream(
                thinking=all_thinking,
                content=all_content,
                tool_calls=tool_calls
            ))

        # 如果存在工具调用，则处理它们
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.function.name == "planning":
                    # 解析参数
                    args = tool_call.function.arguments

                    if isinstance(args, str):
                        args = json.loads(args)

                    # 确保plan_id正确设置并执行工具
                    args["plan_id"] = self.active_plan_id

                    # 执行工具并获取结果
                    result = await self.planning_tool.execute(**args)
                    
                    # 将工具执行结果发送到队列
                    await result_queue.put(AgentResultStream(
                        thinking="",
                        content=str(result),
                        tool_calls=[]
                    ))
                    
                    logger.info(f"计划创建结果: {str(result)}")
                    await result_queue.put(QueueEnd())
                    return

        # 如果执行到达这里，则创建一个默认计划
        logger.warning("正在创建默认计划")

        # Create default plan
        default_result = await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],
            }
        )
        
        # 将默认计划结果发送到队列
        await result_queue.put(AgentResultStream(
            thinking="",
            content=str(default_result),
            tool_calls=[]
        ))
        
        await result_queue.put(QueueEnd())
        return

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        解析当前计划以识别第一个未完成的步骤的索引和信息。
        如果没有活动步骤，则返回 (None, None)。
        """
        plan_data = self._get_plan_data()
        if not plan_data:
            return None, None

        try:
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # 查找第一个未完成的步骤
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # 将当前步骤标记为进行中
                    await self._update_step_status(i, PlanStepStatus.IN_PROGRESS.value)
                    return i, step

            return None, None  # 没有活动步骤

        except Exception as e:
            logger.warning(f"查找当前步骤索引时出错: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict, queue: asyncio.Queue):
        """使用指定的代理执行当前步骤，使用agent.run_stream()。"""
        # 准备当前计划状态的上下文
        plan_status = await self._get_plan_text()
        step_text = step_info.get("step", f"步骤 {self.current_step_index}")

        # 为代理创建一个执行当前步骤的提示
        step_prompt = f"""
        当前计划状态:
        {plan_status}

        你的当前任务:
        你现在正在执行步骤 {self.current_step_index}: "{step_text}"

        请使用适当的工具执行此步骤。请注意你只需要完成当前步骤即可，不需要完成整个计划。完成后，提供你完成的总结。
        """
        step_queue = asyncio.Queue()
        await queue.put(step_queue)

        # 使用agent.run_stream()执行步骤
        try:
            step_result_queue = await executor.run_stream(step_prompt)
            await self._step_result_to_one_step_queue(step_queue, step_result_queue)
            return
        except Exception as e:
            logger.error(f"执行步骤 {self.current_step_index} 时出错: {e}")
            error_queue = asyncio.Queue()
            await error_queue.put(AgentResultStream(
                thinking="",
                content=f"执行步骤 {self.current_step_index} 时出错: {str(e)}",
                tool_calls=[]
            ))
            await error_queue.put(QueueEnd())
            return error_queue

    async def _step_result_to_one_step_queue(self, step_queue: asyncio.Queue, step_result_queue: asyncio.Queue):
        """等待步骤执行完成后标记为已完成"""
        # 等待队列中的所有项目
        while True:
            item = await step_result_queue.get()
            if isinstance(item, AgentDone):
                break
            if isinstance(item, asyncio.Queue):
                while True:
                    result = await item.get()
                    if isinstance(result, QueueEnd):
                        break
                    await step_queue.put(result)
        await step_queue.put(QueueEnd())

        # 标记步骤为已完成
        await self._mark_step_completed()

    async def _mark_step_completed(self) -> None:
        """标记当前步骤为已完成。"""
        if self.current_step_index is None:
            return

        await self._update_step_status(self.current_step_index, PlanStepStatus.COMPLETED.value)

    async def _get_plan_text(self) -> str:
        """获取当前计划作为格式化文本。"""
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"获取计划时出错: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """从存储中直接生成计划文本，如果计划工具失败。"""
        plan_data = self._get_plan_data()
        if not plan_data:
            return f"Error: 计划 ID {self.active_plan_id} 未找到"

        try:
            title = plan_data.get("title", "Untitled Plan")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # 确保step_statuses和step_notes与步骤数量匹配
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # 按状态计数步骤
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            plan_text = f"计划: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += (
                f"进度: {completed}/{total} 步骤已完成 ({progress:.1f}%)\n"
            )
            plan_text += f"状态: {status_counts[PlanStepStatus.COMPLETED.value]} 已完成, {status_counts[PlanStepStatus.IN_PROGRESS.value]} 进行中, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} 阻塞, {status_counts[PlanStepStatus.NOT_STARTED.value]} 未开始\n\n"
            plan_text += "步骤:\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                # 使用状态标记表示步骤状态
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   Notes: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"从存储生成计划文本时出错: {e}")
            return f"Error: 无法检索 ID {self.active_plan_id} 的计划"

    async def _finalize_plan(self, total_queue: asyncio.Queue):
        """完成计划并使用流程的LLM直接提供摘要，以流式方式返回。"""
        plan_text = await self._get_plan_text()
        result_queue = asyncio.Queue()
        await total_queue.put(result_queue)

        try:
            system_message = Message.system_message("你是一个计划助手。你的任务是总结完成的计划。")
            user_message = Message.user_message(
                f"计划已完成。以下是最终计划状态:\n\n{plan_text}\n\n请提供已完成的内容总结和任何最终想法。"
            )

            # 使用流式方式获取LLM响应
            gen = await self.llm.chat(
                messages=[system_message, user_message],
                stream=True,
            )

            all_thinking = ""
            all_content = ""
            
            async for thinking, content, _ in gen:
                all_thinking += thinking
                all_content += content
                
                await result_queue.put(AgentResultStream(
                    thinking=all_thinking,
                    content="计划已完成:\n\n" + all_content if all_content else "",
                    tool_calls=[]
                ))

            await result_queue.put(QueueEnd())
            return
            
        except Exception as e:
            logger.error(f"使用代理总结计划时出错: {e}")
            await result_queue.put(AgentResultStream(
                thinking="",
                content=f"计划已完成。生成总结时出错。\n{e}",
                tool_calls=[]
            ))
            await result_queue.put(QueueEnd())
            return