from core.Agent.assient import AsyncAssistant
import asyncio
from config import *
from core.llms.openai_llm import OpenAICoT

async def main():
    llm = OpenAICoT(
        api_base=LLM_API_BASE,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
    )
    assistant = AsyncAssistant(
        llm=llm,
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
