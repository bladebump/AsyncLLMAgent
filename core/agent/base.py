from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional, AsyncIterator
import asyncio
from utils.log import logger
from core.llms.base import AsyncBaseChatCOTModel
from core.schema import AgentState, AgentDone, Message, AgentResult
from core.mem import AsyncMemory


class BaseAgent(ABC):
    """抽象基类，用于管理代理状态和执行。
    提供状态转换、内存管理以及基于步骤的执行循环的基础功能。子类必须实现 `step` 方法。
    """

    def __init__(
        self,
        name: str,
        llm: AsyncBaseChatCOTModel,
        memory: AsyncMemory,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        next_step_prompt: Optional[str] = None,
        state: AgentState = AgentState.IDLE,
        max_steps: int = 10,
        current_step: int = 0,
        duplicate_threshold: int = 2,
        **kwargs
    ):
        # 核心属性
        self.name = name
        self.description = description

        # 提示
        self.system_prompt = system_prompt
        self.next_step_prompt = next_step_prompt

        # 依赖
        self.llm = llm
        self.memory = memory
        self.state = state

        # 执行控制
        self.max_steps = max_steps
        self.current_step = current_step
        
        self.duplicate_threshold = duplicate_threshold
        self.request = ""
        
        # 允许子类有额外属性
        for key, value in kwargs.items():
            setattr(self, key, value)

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """安全代理状态转换的上下文管理器。"""
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

    async def run(self, request: Optional[str] = None) -> list[AgentResult]:
        """非流式异步执行代理的主循环。"""
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"无法从状态运行代理: {self.state}")

        if request:
            self.request = request
            await self.memory.add(Message.user_message(request))

        results: List[AgentResult] = []
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

                results.append(step_result)

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(AgentResult(reason="", result="终止: 达到最大步骤"))
        return results
    
    async def run_stream(self, request: Optional[str] = None) -> asyncio.Queue:
        """流式异步执行代理的主循环，使用队列返回结果。"""
        result_queue = asyncio.Queue()
        
        if self.state != AgentState.IDLE:
            result_queue.put(AgentDone(reason="代理状态错误"))
            raise RuntimeError(f"无法从状态运行代理: {self.state}")

        if request:
            self.request = request
            await self.memory.add(Message.user_message(request))

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
                await self.step_stream(queue)

                # Check for stuck state
                if await self.is_stuck():
                    await self.handle_stuck_state()

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

    @abstractmethod
    async def step_stream(self, queue: asyncio.Queue):
        """执行代理工作流中的单个步骤，使用流返回结果。
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