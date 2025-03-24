import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.sse import sse_client
from client import deepseek_v3_cot
from core.util import mcp_tool_to_function
import json

class MCPClient:
    def __init__(self):
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = deepseek_v3_cot

    async def connect_to_server(self, url: str):
        """连接到 MCP 服务器

        参数：
            server_script_path: 服务器脚本的路径（.py 或 .js）
        """

        self._streams_context = sse_client(url=url)
        # 将上下文管理器添加到exit_stack中
        streams = await self.exit_stack.enter_async_context(self._streams_context)

        self._session_context = ClientSession(*streams)
        self.session = await self.exit_stack.enter_async_context(self._session_context)

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """使用 OpenAI 和可用工具处理查询"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = mcp_tool_to_function(response.tools)

        response = await self.client.chat_with_functions(messages, available_tools)

        # 处理响应并处理工具调用
        tool_results = []
        final_text = []

        # 检查是否有文本内容
        if response.content:
            final_text.append(response.content)
        
        # 处理可能的工具调用
        if response.tool_calls:
            # 记录助手消息
            assistant_message = {
                "role": "assistant",    
                "content": response.content,
                "tool_calls": response.tool_calls
            }
            messages.append(assistant_message)
            
            # 处理每个工具调用
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                
                try:
                    # 可能需要将字符串解析为JSON
                    if isinstance(tool_args, str):
                        tool_args_obj = json.loads(tool_args)
                    else:
                        tool_args_obj = tool_args
                        
                    # 执行工具调用
                    result = await self.session.call_tool(tool_name, tool_args_obj)
                    tool_results.append({"call": tool_name, "result": result})
                    final_text.append(f"[调用工具 {tool_name}，参数 {tool_args}]")
                    
                    # 添加工具响应到消息历史
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "\n".join([item.model_dump_json() for item in result.content])
                    }
                    messages.append(tool_message)
                except Exception as e:
                    error_message = f"工具调用错误: {str(e)}"
                    final_text.append(error_message)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_message
                    }
                    messages.append(tool_message)
            
            # 获取后续响应
            _, response = await self.client.chat(messages=messages)
            
            final_text.append(response)

        return "\n".join(final_text)

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP 客户端已启动！")
        print("输入您的查询或输入 'quit' 退出。")

        while True:
            try:
                query = input("\n查询：").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\n错误：{str(e)}")

    async def cleanup(self):
        """清理资源"""        
        # 关闭所有由exit_stack管理的上下文
        await self.exit_stack.aclose()
        print("已清理所有资源")


async def main():

    client = MCPClient()
    try:
        await client.connect_to_server("http://localhost:18080/sse")
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())