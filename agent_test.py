from core.agent import ToolCallAgent, SummaryToolCallAgent
import asyncio
from core.config import config
from core.llms.qwen_llm import QwenCoT
from core.tools import GetWeather, RAGTool, MCPClients, Terminate
from core.mem import ListMemory
from core.schema import AgentDone, QueueEnd
from core.embeddings.silicon_agent import SiliconEmbeddingAgent
from core.vector.milvus import MilvusVectorStore
from core.ranks import SiliconRankAgent

async def main():
    llm_config = config.llm_providers["qwen"]
    llm = QwenCoT(
        api_base=llm_config.api_base,
        api_key=llm_config.api_key,
        model=llm_config.model,
        enable_thinking=True
    )
    embedding = SiliconEmbeddingAgent(
        url=config.embedding.api_base,
        api_key=config.embedding.api_key,
        model=config.embedding.model,
    )
    milvus = MilvusVectorStore(
        uri=config.milvus.uri,
        username=config.milvus.username,
        password=config.milvus.password,
        dense_vector_dim=config.milvus.dense_vector_dim,
            use_sparse_vector=config.milvus.use_sparse_vector
        )
    assistant = SummaryToolCallAgent(
        name="法律助手",
        description="一个可以查询法律的助手",
        llm=llm,
        memory=ListMemory(),
    )
    reranker = SiliconRankAgent(
        url=config.reranker.api_base,
        api_key=config.reranker.api_key,
        model=config.reranker.model,
    )
    # mcp = MCPClients()
    # await mcp.connect_sse("https://mcp-8b5eddad-053e-451b.api-inference.modelscope.cn/sse")
    assistant.available_tools.add_tool(RAGTool(
        description="一个可以查询法律的工具，会去检索相关文档，并返回检索到的文档内容，只有当用户的问题与法律相关时，才使用这个工具",
        milvus_client=milvus,
        collection_name="law",
        llm=llm,
        text_embedder=embedding,
        reranker=reranker
    ))
    queue = await assistant.run_stream("你有哪些函数可用")
    
    while True:
        chunk = await queue.get()
        if isinstance(chunk, AgentDone):
            break
        if isinstance(chunk, asyncio.Queue):
            while True:
                result = await chunk.get()
                if isinstance(result, QueueEnd):
                    break
                if result.thinking:
                    print(result.thinking)
                if result.content:
                    print(result.content)
                if result.tool_calls:
                    print(result.tool_calls)
if __name__ == "__main__":
    asyncio.run(main())