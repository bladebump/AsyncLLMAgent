from agents import Agent, Runner, RunConfig, OpenAIProvider
from openai import AsyncOpenAI

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

client = AsyncOpenAI(
    api_key="sk-ikrvbgsuezjiomgtczsggqgwwuexjvaxksdabgkcknejbklg",
    base_url="https://api.siliconflow.cn/v1",
)

model_provider = OpenAIProvider(
    openai_client=client,
    use_responses=False
)

run_config = RunConfig(
    model_provider=model_provider,
    model="Pro/deepseek-ai/DeepSeek-V3",
)

result = Runner.run_sync(agent, "为什么人总是怀念小时候", run_config=run_config)
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.