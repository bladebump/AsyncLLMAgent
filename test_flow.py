import asyncio
from core.llms.qwen_llm import QwenCoT
from core.config import config
from core.flow import PlanningFlow
from core.agent import ToolCallAgent
from core.mem import ListMemory
from core.tools import Bash

async def main():
    llm_config = config.llm_providers["qwen"]
    llm = QwenCoT(
        api_base=llm_config.api_base,
        api_key=llm_config.api_key,
        model=llm_config.model,
        enable_thinking=True
    )
    agent = ToolCallAgent(
        name="命令行执行专家",
        description="一个擅长执行命令行任务的专家,当前系统是linux系统",
        llm=llm,
        memory=ListMemory(),
    )
    agent.available_tools.add_tool(Bash())
    flow = PlanningFlow(
        llm=llm,
        agents=[agent]
    )
    try:
        result = await asyncio.wait_for(
            flow.execute(r"如何才能将/root/文件夹中的所有文件名打印出来"),
            timeout=3600
        )
        for r in result:
            print("--------------------------------")
            print(r.thinking)
            print("--------------------------------")
            print(r.content)
    except asyncio.TimeoutError:
        print("执行超时")

if __name__ == "__main__":
    asyncio.run(main())
