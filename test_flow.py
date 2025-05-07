import asyncio
from core.llms.qwen_llm import QwenCoT
from core.config import config

async def main():
    llm_config = config.llm_providers["qwen"]
    llm = QwenCoT(
        api_base=llm_config.api_base,
        api_key=llm_config.api_key,
        model=llm_config.model,
        enable_thinking=True
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    }
                }   
            }
        }
    ]
    response = await llm.chat(messages=[
        {"role": "user", "content": "你好，我想知道北京今天的天气怎么样？"}
    ], tools = tools, stream=False)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
