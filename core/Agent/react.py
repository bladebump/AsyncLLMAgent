from abc import ABC, abstractmethod
from core.agent.base import BaseAgent


class ReActAgent(BaseAgent, ABC):
    """ReAct模式的代理基类，实现"思考-行动"循环"""

    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""

    async def step(self) -> str:
        """Execute a single step: think and act."""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()