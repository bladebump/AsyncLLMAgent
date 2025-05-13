import multiprocessing
import uvicorn
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from starlette.routing import Mount
from starlette.applications import Starlette
from mcp.types import TextContent


def create_app(server_name: str):
    """构建单个 Starlette SSE App"""
    mcp = FastMCP(server_name)
    api = FastAPI(
        title=f"{server_name} API",
        version="1.0.0",
        description=f"{server_name} 提供的测试服务"
    )

    @api.get("/")
    async def index():
        return {"message": f"Welcome to {server_name}"}

    @mcp.tool()
    @api.get("/status")
    def get_status() -> TextContent:
    
        return TextContent(
            type="text",  
            text=f"{server_name} 被成功调用"
        )

    return Starlette(
        routes=[
            Mount('/api', app=api),
            Mount('/', app=mcp.sse_app()),
        ]
    )


def run_server(port: int, name: str):
    """启动单个 Uvicorn 进程"""
    app = create_app(name)
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False, log_level="info")


if __name__ == "__main__":
    servers = [
        {"port": 5001, "name": "server1"},
        {"port": 5002, "name": "server2"},
        {"port": 5003, "name": "server3"},
    ]

    processes = []
    for config in servers:
        p = multiprocessing.Process(target=run_server, args=(config["port"], config["name"]), daemon=True)
        p.start()
        processes.append(p)

    print("多个SSE服务已启动，按 Ctrl+C 停止。")
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("正在关闭所有服务器")