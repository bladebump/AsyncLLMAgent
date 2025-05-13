from asyncio.queues import Queue
from typing import Any, Dict, List, Optional, Tuple
from pydantic import Field
from core.agent import ToolCallAgent
from core.agent.toolcall import SYSTEM_PROMPT
from core.llms.base import AsyncBaseChatCOTModel
from core.mem.base import AsyncMemory
from core.tools.tool_collection import ToolCollection
from utils.log import logger
from core.schema import AgentState, Message, ToolChoice
from core.tools import MCPClients

SYSTEM_PROMPT = """你是一个可以访问模型上下文协议(MCP)服务器的AI助手。
你可以使用MCP服务器提供的工具来完成任务。
MCP服务器会动态提供你可以使用的工具 - 请始终先检查可用的工具。

使用MCP工具时：
1. 根据任务需求选择适当的工具
2. 按照工具要求提供格式正确的参数
3. 观察结果并用它们来确定下一步操作
4. 工具可能在操作过程中变化 - 新工具可能出现或现有工具可能消失

请遵循以下指导原则：
- 使用工具时提供其模式文档中要求的有效参数
- 通过理解错误原因并使用更正的参数重试来优雅地处理错误
- 对于多媒体响应(如图像)，你将收到内容的描述
- 逐步完成用户请求，使用最合适的工具
- 如果需要按顺序调用多个工具，请一次调用一个并等待结果

记得向用户清晰地解释你的推理和行动。
"""

class MCPAgent(ToolCallAgent):
    """用于与MCP(模型上下文协议)服务器交互的代理。

    此代理使用SSE或stdio传输连接到MCP服务器，
    并通过代理的工具接口使服务器的工具可用。
    """

    def __init__(self, name: str = "mcp_agent", 
                 llm: AsyncBaseChatCOTModel = None, 
                 memory: AsyncMemory = None, 
                 description: str = "一个可以访问模型上下文协议(MCP)服务器和使用其工具的AI助手。", 
                 system_prompt: str = SYSTEM_PROMPT, 
                 state: AgentState = AgentState.IDLE, 
                 available_tools: ToolCollection | None = None, 
                 tool_choices: str = ToolChoice.AUTO, 
                 max_steps: int = 30, 
                 max_observe: int | bool | None = None, 
                 connection_type: str = "sse",
                 **kwargs):
        super().__init__(name, llm, memory, description, system_prompt, state, available_tools, tool_choices, max_steps, max_observe, **kwargs)
        self.mcp_clients = MCPClients()
        self.connection_type = connection_type


    async def initialize(
        self,
        connection_type: Optional[str] = None,
        server_url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
    ) -> None:
        """初始化MCP连接。

        参数:
            connection_type: 要使用的连接类型("stdio"或"sse")
            server_url: MCP服务器的URL(用于SSE连接)
            command: 要运行的命令(用于stdio连接)
            args: 命令的参数(用于stdio连接)
        """
        if connection_type:
            self.connection_type = connection_type

        # 根据连接类型连接到MCP服务器
        if self.connection_type == "sse":
            if not server_url:
                raise ValueError("SSE连接需要服务器URL")
            await self.mcp_clients.connect_sse(server_url=server_url)
        elif self.connection_type == "stdio":
            if not command:
                raise ValueError("stdio连接需要命令")
            await self.mcp_clients.connect_stdio(command=command, args=args or [])
        else:
            raise ValueError(f"不支持的连接类型: {self.connection_type}")

        # 将available_tools设置为我们的MCP实例
        self.available_tools = self.mcp_clients

        # 存储初始工具模式
        await self._refresh_tools()

        # 添加关于可用工具的系统消息
        tool_names = list(self.mcp_clients.tool_map.keys())
        tools_info = ", ".join(tool_names)

        # 添加系统提示和可用工具信息
        self.memory.add_message(
            Message.system_message(
                f"{self.system_prompt}\n\n可用的MCP工具: {tools_info}"
            )
        )

    async def _refresh_tools(self) -> Tuple[List[str], List[str]]:
        """从MCP服务器刷新可用工具列表。"""
        if not self.mcp_clients.session:
            return [], []

        # 直接从服务器获取当前工具模式
        response = await self.mcp_clients.list_tools()
        current_tools = {tool.name: tool.inputSchema for tool in response.tools}

        # 确定添加、删除和更改的工具
        current_names = set(current_tools.keys())
        previous_names = set(self.tool_schemas.keys())

        added_tools = list(current_names - previous_names)
        removed_tools = list(previous_names - current_names)

        # 检查现有工具的模式变化
        changed_tools = []
        for name in current_names.intersection(previous_names):
            if current_tools[name] != self.tool_schemas.get(name):
                changed_tools.append(name)

        # 更新存储的模式
        self.tool_schemas = current_tools

        # 记录并通知变化
        if added_tools:
            logger.info(f"添加了MCP工具: {added_tools}")
            self.memory.add_message(
                Message.system_message(f"新工具可用: {', '.join(added_tools)}")
            )
        if removed_tools:
            logger.info(f"移除了MCP工具: {removed_tools}")
            self.memory.add_message(
                Message.system_message(
                    f"工具不再可用: {', '.join(removed_tools)}"
                )
            )
        if changed_tools:
            logger.info(f"更改了MCP工具: {changed_tools}")

        return added_tools, removed_tools

    async def cleanup(self) -> None:
        """完成后清理MCP连接。"""
        if self.mcp_clients.session:
            await self.mcp_clients.disconnect_all()
            logger.info("MCP连接已关闭")