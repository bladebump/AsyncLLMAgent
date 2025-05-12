from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from core.agent import BaseAgent
from core.schema import AgentResult, AgentResultStream
import asyncio
class BaseFlow(ABC):
    """支持多个代理的执行流程基类"""

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], tools: Optional[List] = None, primary_agent_key: Optional[str] = None
    ):
        if isinstance(agents, BaseAgent):
            agents_dict = {agents.name: agents}
        elif isinstance(agents, list):
            agents_dict = {agent.name: agent for agent in agents}
        else:
            agents_dict = agents

        self.agents = agents_dict
        self.tools = tools
        if primary_agent_key is None:
            self.primary_agent_key = list(agents_dict.keys())[0]
        else:
            self.primary_agent_key = primary_agent_key

    @property
    def primary_agent(self) -> Optional[BaseAgent]:
        """获取流程的主代理"""
        return self.agents.get(self.primary_agent_key)

    def get_agent(self, key: str) -> Optional[BaseAgent]:
        """获取指定代理"""
        return self.agents.get(key)

    def add_agent(self, key: str, agent: BaseAgent) -> None:
        """添加新的代理"""
        self.agents[key] = agent

    @abstractmethod
    async def execute(self, input_text: str) -> asyncio.Queue:
        """执行流程，以流式方式返回结果"""