from abc import ABC, abstractmethod
from asyncio import Queue
from core.agent.base import BaseAgent
from core.schema import AgentResult, QueueEnd
import asyncio

class ReActAgent(BaseAgent, ABC):
    """ReAct模式的代理基类，实现"思考-行动"循环"""

    @abstractmethod
    async def think(self) -> tuple[str, str, bool]:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""
    
    @abstractmethod
    async def think_stream(self):
        """流式返回的think"""
    
    @abstractmethod
    async def act_stream(self):
        """流式返回的act"""

    async def step(self) -> AgentResult:
        """Execute a single step: think and act."""
        thinking, content, should_act = await self.think()
        if not should_act:
            return AgentResult(thinking=thinking, content=content)
        result = await self.act()
        return AgentResult(thinking=thinking, content=result)
    
    async def step_stream(self, queue: Queue):
        """Execute a single step: think and act with stream"""
        # 直接执行think_stream方法并将结果放入队列
        think_queue = asyncio.Queue()
        await queue.put(think_queue)
        think_gen = self.think_stream()
        async for result in think_gen:
            await think_queue.put(result)
        await think_queue.put(QueueEnd())
            
        # 直接执行act_stream方法并将结果放入队列
        act_queue = asyncio.Queue()
        await queue.put(act_queue)
        act_gen = self.act_stream()
        async for result in act_gen:
            await act_queue.put(result)
        await act_queue.put(QueueEnd())
