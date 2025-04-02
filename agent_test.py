from core.Agent.toolcall import ToolCallAgent
import asyncio
from config import *
from core.llms.openai_llm import OpenAICoT
from core.tools import GetWeather
from core.mem import ListMemory

async def main():
    llm = OpenAICoT(
        api_base=LLM_API_BASE,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
    )
    assistant = ToolCallAgent(
        name="天气助手",
        description="一个可以获取天气信息的助手",
        llm=llm,
        memory=ListMemory(),
    )
    assistant.available_tools.add_tool(GetWeather())
    response = await assistant.run("杭州的天气怎么样？")
    print(response)
    
if __name__ == "__main__":
    asyncio.run(main())
