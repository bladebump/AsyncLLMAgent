from contextlib import AsyncExitStack
from typing import List, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from utils.log import logger
from core.tools.base import BaseTool, ToolResult
from core.tools.tool_collection import ToolCollection



class MCPClientTool(BaseTool):
    """表示一个工具代理，可以远程调用MCP服务器上的工具。"""

    session: Optional[ClientSession] = None

    async def execute(self, **kwargs) -> ToolResult:
        """通过远程调用MCP服务器执行工具。"""
        if not self.session:
            return ToolResult(error="未连接到MCP服务器")

        try:
            result = await self.session.call_tool(self.name, kwargs)
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "未返回输出。")
        except Exception as e:
            return ToolResult(error=f"执行工具时出错: {str(e)}")



class MCPClients(ToolCollection):
    """
    一个工具集合，连接多个MCP服务器并管理可用工具。
    """

    def __init__(self):
        super().__init__()
        self.name = "mcp"
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = {}
        self.description = "MCP客户端工具用于多个服务器交互"


    async def connect_sse(self, server_id: str, server_url: str) -> None:
        if not server_url or not server_id:
            raise ValueError("服务器ID和URL是必需的。")
        await self.disconnect(server_id)

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        streams_context = sse_client(url=server_url)
        streams = await exit_stack.enter_async_context(streams_context)
        read, write = streams
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        self.sessions[server_id] = session

        await self._initialize_and_list_tools(server_id, session)

    async def connect_stdio(self, command: str, args: List[str]) -> None:
        """使用stdio传输连接到MCP服务器。"""
        if not command:
            raise ValueError("服务器命令是必需的。")
        if self.session:
            await self.disconnect()

        server_params = StdioServerParameters(command=command, args=args)
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self._initialize_and_list_tools()


    async def _initialize_and_list_tools(self, server_id: str, session: ClientSession) -> None:
        """初始化会话并填充工具映射。"""
        await session.initialize()
        response = await session.list_tools()

        for tool in response.tools:
            qualified_name = f"{server_id}:{tool.name}"
            server_tool = MCPClientTool(
                name=qualified_name,
                description=f"[{server_id}] {tool.description}",
                parameters=tool.inputSchema,
                session=session,
            )
            self.tool_map[qualified_name] = server_tool

        self.tools = tuple(self.tool_map.values())
        logger.info(f"[{server_id}] 已连接，工具: {[tool.name for tool in response.tools]}")

    async def disconnect(self, server_id: str) -> None:
        """断开与MCP服务器的连接并清理资源。"""
        if server_id in self.exit_stacks:
            try:
                await self.exit_stacks[server_id].aclose()
            except Exception as e:
                logger.warning(f"关闭 {server_id} 时发生异常: {e}")
            finally:
                del self.exit_stacks[server_id]
        if server_id in self.sessions:
            del self.sessions[server_id]
            
        self.tool_map = {
            name: tool for name, tool in self.tool_map.items()
            if not name.startswith(f"{server_id}:")
        }
        self.tools = tuple(self.tool_map.values())
        logger.info(f"[{server_id}] 已断开连接")
