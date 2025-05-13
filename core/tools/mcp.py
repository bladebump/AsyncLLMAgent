from contextlib import AsyncExitStack
from typing import List, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, ListToolsResult

from utils.log import logger
from core.tools.base import BaseTool, ToolResult
from core.tools.tool_collection import ToolCollection



class MCPClientTool(BaseTool):
    """表示一个工具代理，可以远程调用MCP服务器上的工具。"""

    session: Optional[ClientSession] = None
    server_id: str
    original_name: str

    async def execute(self, **kwargs) -> ToolResult:
        """通过远程调用MCP服务器执行工具。"""
        if not self.session:
            return ToolResult(error="未连接到MCP服务器")

        try:
            logger.info(f"[{self.server_id}] 执行工具: {self.original_name}")
            result = await self.session.call_tool(self.original_name, kwargs)
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
        
        if server_id in self.exit_stacks:
            await self.disconnect(server_id)

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        streams_context = sse_client(url=server_url)
        streams = await exit_stack.enter_async_context(streams_context)
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        self.sessions[server_id] = session

        await self._initialize_and_list_tools(server_id)

    async def connect_stdio(self, command: str, args: List[str], server_id: str) -> None:
        """使用stdio传输连接到MCP服务器。"""
        if not command or not server_id:
            raise ValueError("服务器命令和ID是必需的。")
        if server_id in self.sessions:
            await self.disconnect(server_id)

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        server_params = StdioServerParameters(command=command, args=args)
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        self.sessions[server_id] = session

        await self._initialize_and_list_tools(server_id)


    async def _initialize_and_list_tools(self, server_id: str) -> None:
        """初始化会话并填充工具映射。"""
        session = self.sessions.get(server_id)
        if not session:
            raise ValueError(f"服务器 {server_id} 未连接。")
        await session.initialize()
        response = await session.list_tools()

        for tool in response.tools:
            original_name = tool.name
            tool_name = f"mcp_{server_id}_{original_name}"
            server_tool = MCPClientTool(
                name=tool_name,
                description=f"[{server_id}] {tool.description}",
                parameters=tool.inputSchema,
                session=session,
                server_id=server_id,
                original_name=original_name,
            )
            self.tool_map[tool_name] = server_tool

        self.tools = tuple(self.tool_map.values())
        logger.info(f"[{server_id}] 已连接，工具: {[tool.name for tool in response.tools]}")

    async def list_tools(self) -> ListToolsResult:
        """列出所有工具。"""
        tools_result = ListToolsResult(tools=[])
        for session in self.sessions.values():
            response = await session.list_tools()
            tools_result.tools += response.tools
        return tools_result

    async def disconnect(self, server_id: str) -> None:
        """断开与MCP服务器的连接并清理资源。"""
        if server_id in self.exit_stacks:
            try:
                exit_stack = self.exit_stacks.get(server_id)
                if exit_stack:
                    await exit_stack.aclose()
            except Exception as e:
                logger.warning(f"关闭 {server_id} 时发生异常: {e}")
            finally:
                self.sessions.pop(server_id, None)
                self.exit_stacks.pop(server_id, None)

        self.tool_map = {k:v for k,v in self.tool_map.items() if v.server_id != server_id }
        self.tools = tuple(self.tool_map.values())
        logger.info(f"[{server_id}] 已断开连接")
    
    async def disconnect_all(self) -> None:
        """断开所有MCP服务器连接。"""
        for server_id in list(self.exit_stacks.keys()):
            await self.disconnect(server_id)
