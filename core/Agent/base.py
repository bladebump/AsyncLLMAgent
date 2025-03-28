from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional, AsyncGenerator
import asyncio
from pydantic import BaseModel, Field, model_validator
from core.llms.base import AsyncBaseChatCOTModel
from utils.log import logger
from core.schema import ROLE_TYPE, AgentState, AgentDone, Message, AgentResult
from core.mem import AsyncMemory


class BaseAgent(BaseModel, ABC):
    """抽象基类，用于管理代理状态和执行。
    提供状态转换、内存管理以及基于步骤的执行循环的基础功能。子类必须实现 `step` 方法。
    """

    # 核心属性
    name: str = Field(..., description="代理的唯一名称")
    description: Optional[str] = Field(None, description="可选的代理描述")

    # 提示
    system_prompt: Optional[str] = Field(
        None, description="系统级别的指令提示"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="确定下一步操作的提示"
    )

    # 依赖
    llm: AsyncBaseChatCOTModel = Field(..., description="语言模型实例")
    memory: AsyncMemory = Field(..., description="代理的内存存储")
    state: AgentState = Field(
        default=AgentState.IDLE, description="当前代理状态"
    )

    # 执行控制
    max_steps: int = Field(default=10, description="最大步骤数")
    current_step: int = Field(default=0, description="当前步骤")

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # 允许子类有额外字段

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """安全代理状态转换的上下文管理器。

        Args:
            new_state: 在上下文期间要转换到的状态。

        Yields:
            None: 允许在新的状态下执行。

        Raises:
            ValueError: 如果new_state无效。
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # 在失败时转换为ERROR
            raise e
        finally:
            self.state = previous_state  # 恢复到之前的状态

    async def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """将消息添加到代理的内存中。

        Args:
            role: 消息发送者的角色（用户、系统、助手、工具）。
            content: 消息内容。
            base64_image: 可选的base64编码图像。
            **kwargs: 其他参数（例如工具调用ID用于工具消息）。

        Raises:
            ValueError: 如果role不支持。
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"不支持的消息角色: {role}")

        # 根据角色创建消息，使用适当的参数
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        await self.memory.add(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> AgentResult:
        """异步执行代理的主循环。

        Args:
            request: 可选的初始用户请求。

        Returns:
            总结执行结果的字符串。

        Raises:
            RuntimeError: 如果代理在开始时不是IDLE状态。
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"无法从状态运行代理: {self.state}")

        if request:
            await self.update_memory("user", request)

        results: List[str] = []
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

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"终止: 达到最大步骤 ({self.max_steps})")
        return "\n".join(results) if results else "未执行步骤"
    
    async def run_stream(self, request: Optional[str] = None) -> asyncio.Queue:
        """异步执行代理的主循环，使用队列返回结果。

        Args:
            request: 可选的初始用户请求。

        Returns:
            asyncio.Queue: 包含结果的队列，可以通过queue.get()获取结果。
                           最后会返回一个AgentDone对象标记执行完成。

        Raises:
            RuntimeError: 如果代理在开始时不是IDLE状态。
        """
        result_queue = asyncio.Queue()
        
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"无法从状态运行代理: {self.state}")

        if request:
            await self.update_memory("user", request)

        # 在后台任务中执行，避免阻塞
        asyncio.create_task(self._run_and_fill_queue(result_queue))
        
        return result_queue

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
                
            # 标记队列已完成
            await queue.put(AgentDone(reason="执行完成"))

    @abstractmethod
    async def step(self) -> AgentResult:
        """执行代理工作流中的单个步骤。
        必须由子类实现以定义特定行为。
        """

    async def handle_stuck_state(self):
        """通过添加提示来处理卡住状态"""
        stuck_prompt = "\n观察到重复的响应。考虑新的策略并避免重复已经尝试过的无效路径。"
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"代理检测到卡住状态。添加提示: {stuck_prompt}")

    async def is_stuck(self) -> bool:
        """检查代理是否在循环中卡住，通过检测重复的内容"""
        if len(self.memory) < 2:
            return False

        last_message = self.memory.Messages[-1]
        if not last_message.content:
            return False

        # 计算相同内容的出现次数
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.Messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    async def messages(self) -> List[Message]:
        """从代理的内存中检索消息列表。"""
        return self.memory.Messages

    @messages.setter
    async def messages(self, value: List[Message]):
        """设置代理的内存中的消息列表。"""
        self.memory.Messages = value