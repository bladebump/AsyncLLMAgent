import asyncio
from core.llms.qwen_llm import QwenCoT
from core.config import config
from core.flow import PlanningFlow
from core.agent import ToolCallAgent
from core.mem import ListMemory
from core.tools import Bash,GetWeather
from core.schema import AgentResultStream, AgentDone, QueueEnd

async def main():
    llm_config = config.llm_providers["qwen"]
    llm = QwenCoT(
        api_base=llm_config.api_base,
        api_key=llm_config.api_key,
        model=llm_config.model,
        enable_thinking=True
    )
    agent = ToolCallAgent(
        name="旅游规划师",
        description="一个擅长规划旅游路线的专家",
        llm=llm,
        memory=ListMemory(),
    )
    agent.available_tools.add_tool(GetWeather())
    flow = PlanningFlow(
        llm=llm,
        agents=[agent]
    )
    try:
        queue = await asyncio.wait_for(
            flow.execute(r"规划杭州去北京旅游的路线"),
            timeout=3600
        )
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
    except asyncio.TimeoutError:
        print("执行超时")

if __name__ == "__main__":
    asyncio.run(main())
