from core.agent import ToolCallAgent
import asyncio
from core.config import config
from core.llms.qwen_llm import QwenCoT
from core.tools import GetWeather, RAGTool, MCPClients, PlanningTool
from core.mem import ListMemory
from core.schema import AgentDone, QueueEnd, Message

async def main():
    llm_config = config.llm_providers["qwen"]
    llm = QwenCoT(
        api_base=llm_config.api_base,
        api_key=llm_config.api_key,
        model=llm_config.model,
        enable_thinking=True
    )
    assistant = ToolCallAgent(
        name="计划生成助手",
        description="一个可以生成计划的助手",
        llm=llm,
        memory=ListMemory(),
        system_prompt = (
            "你是一个计划助手。创建一个简洁、可操作的计划，具有清晰的步骤。 "
            "专注于关键里程碑，而不是详细的子步骤。 "
            "优化清晰度和效率。"
            "你只需要创建计划即可，不需要执行计划。"
        )
    )
    # mcp = MCPClients()
    # await mcp.connect_sse("https://mcp-8b5eddad-053e-451b.api-inference.modelscope.cn/sse")
    assistant.available_tools.add_tool(PlanningTool())
    queue = await assistant.run_stream("创建一个合理的计划，具有清晰的步骤，以完成任务: 从杭州出发到重庆的十一假期旅游计划")
    
    while True:
        chunk = await queue.get()
        if isinstance(chunk, AgentDone):
            break
        if isinstance(chunk, asyncio.Queue):
            while True:
                result = await chunk.get()
                if isinstance(result, QueueEnd):
                    break
                if result.thinking:
                    print("thinking:", result.thinking)
                if result.content:
                    print("content:", result.content)
                if result.tool_calls:
                    print("tool_calls:", result.tool_calls)
if __name__ == "__main__":
    asyncio.run(main())