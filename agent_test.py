from core.Agent.assient import AsyncAssistant
from client import deepseek_v3_cot, deepseek_r1_cot
import asyncio

async def main():
    assistant = AsyncAssistant(
        llm=deepseek_r1_cot,
        memory=None,
        function_list=[],
        instruction="以懂王特朗普的口吻回答问题",
        name="Assistant",
        stream=True
    )
    response = await assistant.run("如何看待LGBTQ")
    async for thinking, content in response:
        if thinking:
            print(thinking)
        if content:
            print(content)
    
if __name__ == "__main__":
    asyncio.run(main())
