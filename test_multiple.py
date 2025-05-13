import asyncio
from core.tools.mcp import MCPClients
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_connect_multiple_sse():
    mcp_clients = MCPClients()
    try:
        await mcp_clients.connect_sse("server1", "http://127.0.0.1:5001/sse")
        await mcp_clients.connect_sse("server2", "http://127.0.0.1:5002/sse")
        await mcp_clients.connect_sse("server3", "http://127.0.0.1:5003/sse")

        logger.info("=== 注册的工具 ===")
        for tool in mcp_clients.tools:
            logger.info(f"工具: {tool.name}")

        logger.info("=== 测试调用 ===")
        for server_id in ["server1", "server2", "server3"]:
            tool_key = f"{server_id}:get_status"
            tool = mcp_clients.tool_map[tool_key]
            tool.name = "get_status"  
            result = await tool.execute()
            logger.info(f"[{server_id}] 返回内容: {result.output}")


        for server_id in ["server1", "server2", "server3"]:
            tool_key = f"{server_id}:get_status"            
            result = await tool.execute()
            logger.info(f"[{server_id}] 返回内容: {result.output}")

    finally:
        for server_id in ["server1", "server2", "server3"]:
            await mcp_clients.disconnect(server_id)


if __name__ == "__main__":
    try:
        asyncio.run(test_connect_multiple_sse())
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试执行失败: {str(e)}", exc_info=True)