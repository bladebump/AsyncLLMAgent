from client import embedding_agent
import asyncio
async def test_embedding_agent():
    embedding = await embedding_agent.encode("你好")
    print(embedding)

if __name__ == "__main__":
    asyncio.run(test_embedding_agent())
