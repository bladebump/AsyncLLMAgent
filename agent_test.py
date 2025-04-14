from core.agent.toolcall import ToolCallAgent
import asyncio
from core.config import config
from core.llms import OpenAICoT
from core.tools.get_weather import GetWeather
from core.mem import ListMemory
from core.schema import AgentDone

async def main():
    llm_provider = config.current_provider
    llm_config = config.llm_providers[llm_provider]
    llm = OpenAICoT(
        api_base=llm_config.api_base,
        api_key=llm_config.api_key,
        model=llm_config.model,
    )
    assistant = ToolCallAgent(
        name="天气助手",
        description="一个可以查询天气的助手",
        llm=llm,
        memory=ListMemory(),
    )
    assistant.available_tools.add_tool(GetWeather())
    queue = await assistant.run_stream("杭州的天气如何")
    
    while True:
        chunk = await queue.get()
        if isinstance(chunk, AgentDone):
            break
        print(chunk)
    print(assistant.memory)
    
if __name__ == "__main__":
    asyncio.run(main())
