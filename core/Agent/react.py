from abc import ABC, abstractmethod
from core.agent.base import BaseAgent
from core.schema import AgentResult


class ReActAgent(BaseAgent, ABC):
    """ReAct模式的代理基类，实现"思考-行动"循环"""

    @abstractmethod
    async def think(self) -> str:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""

    async def step(self) -> AgentResult:
        """Execute a single step: think and act."""
        should_act = await self.think()
        if not should_act:
            return AgentResult(reason="", result=should_act)
        result = await self.act()
        return AgentResult(reason=should_act, result=result)