from core.tools.base import BaseTool
from pydantic import Field
from core.vector.milvus import MilvusVectorStore
from core.rags import QuestionRag
from core.llms import AsyncBaseChatCOTModel
from core.embeddings import EmbeddingAgent
from core.ranks import AsyncRankAgent

class RAGTool(BaseTool):
    """一个用于执行RAG任务的工具"""

    name: str = "rag"
    description: str = "一个用于执行RAG任务的工具，会去检索相关文档，并返回检索到的文档内容"
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "查询内容"}
        },
        "required": ["query"]
    }

    milvus_client: MilvusVectorStore = Field(...)
    collection_name: str = Field(...)
    llm: AsyncBaseChatCOTModel = Field(...)
    text_embedder: EmbeddingAgent = Field(...)
    reranker: AsyncRankAgent = Field(default=None)

    async def execute(self, query: str) -> str:
        """执行RAG任务"""
        rag = QuestionRag(query=query, 
                          collection_name=self.collection_name, 
                          text_embedder=self.text_embedder, 
                          vector_store=self.milvus_client, 
                          llm=self.llm, 
                          reranker=self.reranker)
        docs = await rag.choose_doc_for_answer()
        docs_str = ""
        for doc in docs:
            docs_str += f"-----------------------------------\n"
            docs_str += f"文件名：{doc.filename}\n"
            docs_str += f"内容：{doc.text}\n"
            docs_str += f"-----------------------------------\n"
        return docs_str
