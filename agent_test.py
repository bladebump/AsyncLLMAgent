from core.agent.toolcall import ToolCallAgent
import asyncio
from core.config import config
from core.llms import OpenAICoT
from core.tools import PowerShell
from core.mem import ListMemory
from core.schema import AgentDone

async def main():
    llm = OpenAICoT(
        api_base=config.llm.api_base,
        api_key=config.llm.api_key,
        model=config.llm.model,
    )
    assistant = ToolCallAgent(
        name="命令行助手",
        description="一个可以执行命令行命令的助手",
        llm=llm,
        memory=ListMemory(),
    )
    assistant.available_tools.add_tool(PowerShell())
    queue = await assistant.run_stream("查看一下我这周的git提交记录，并总结成周报")
    
    while True:
        chunk = await queue.get()
        if isinstance(chunk, AgentDone):
            break
        print(chunk)
    
if __name__ == "__main__":
    powershell = PowerShell()
    asyncio.run(powershell.execute("Get-ChildItem"))
    # asyncio.run(main())
