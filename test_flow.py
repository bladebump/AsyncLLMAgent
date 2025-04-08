import asyncio
from core.flow import PlanningFlow
from core.agent import ToolCallAgent
from core.llms import OpenAICoT
from core.config import config
from core.mem import ListMemory
from core.tools import Bash

async def main():
    llm = OpenAICoT(
        api_base=config.llm.api_base,
        api_key=config.llm.api_key,
        model=config.llm.model,
    )
    agent = ToolCallAgent(
        llm=llm,
        memory=ListMemory(),
    )
    agent.available_tools.add(Bash())
    flow = PlanningFlow(
        llm=llm,
        agents=[agent]
    )
    try:
        result = await asyncio.wait_for(
            flow.execute("如何才能将一个文件夹中的所有文件名打印出来"),
            timeout=3600
        )
        print(result)
    except asyncio.TimeoutError:
        print("执行超时")

if __name__ == "__main__":
    asyncio.run(main())
