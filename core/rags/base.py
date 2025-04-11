from abc import ABC, abstractmethod
from core.llms.base import AsyncBaseChatCOTModel
from core.vector.base import VectorStoreBase, Document
from core.embeddings.base import EmbeddingAgent
from core.ranks.base import AsyncRankAgent
import re
from typing import List, Dict, Tuple, Any, Optional
from utils.log import logger
from core.config import config

class BaseRag(ABC):
    """基础RAG类"""
    def __init__(self, query: str, 
                 collection_name: str, 
                 text_embedder: EmbeddingAgent, 
                 vector_store: VectorStoreBase, 
                 department: List[str] | None = None, 
                 messages: List[Any] | None = None,  
                 llm: AsyncBaseChatCOTModel | None = None,
                 reranker: AsyncRankAgent | None = None):
        self.query = query
        self.collection_name = collection_name
        self.department = department
        self.messages = messages.copy() if messages else []
        self.text_embedder = text_embedder
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker
    
    @abstractmethod
    async def search_for_docs(self) -> List[Document]:
        """搜索文档

        Args:
            collection_name (str): 文档集合名称
            department (list): 部门列表

        Returns:
            List[Document]: 文档列表
        """
        raise NotImplementedError
    
    async def choose_doc_for_answer(self, top_k: int = config.rag.top_k) -> List[Document]:
        """选择文档作为答案

        Args:
            top_k (int): 返回的文档数量

        Returns:
            List[Document]: 最相关的文档列表
        """
        if self.collection_name is None or self.vector_store is None:
            return []
        
        try:
            self.docs = await self.search_for_docs()
        except Exception as e:
            self.docs = []
            logger.error(f"search for docs error: {str(e)}")

        if len(self.docs) == 0:
            return []
            
        # 如果有rerank功能，可以进行重排序
        if self.reranker:
            try:
                passages = [doc.text for doc in self.docs]
                rerank_result = await self.reranker.rerank(query=self.query, passages=passages)
                rerank_ids = rerank_result['rerank_ids'][:top_k]
                self.docs = [self.docs[i] for i in rerank_ids]
            except Exception as e:
                logger.error(f"rerank error: {str(e)}")
            
        return self.docs[:top_k]
    
    async def search_vector(self, query_list: List[str], qa_thould: float = 0.5) -> List[Document]:
        """搜索向量存储中的文档

        Args:
            query_list (List[str]): 查询列表

        Returns:
            List[Document]: 文档列表
        """
        query_embedding = await self.text_embedder.encode(query_list, task_type="retrieval.query")

        # 准备过滤条件
        filter = None
        if self.department and len(self.department) > 0:
            filter = {"department": self.department}
        
        docs = await self.vector_store.vector_search(
            dense_vector=query_embedding,
            limit=50,
            filter=filter,
            collection_name=self.collection_name,
            anns_field="dense_vector"
        )

        # 按照text去重，保持文档的唯一性
        unique_docs = []
        seen_texts = set()
        for doc in docs:
            hash_value = hash(doc.text)
            if hash_value not in seen_texts:
                seen_texts.add(hash_value)
                unique_docs.append(doc)
        return unique_docs

    async def query_vector(self, keywords: List[str]) -> List[Document]:
        """关键词查询向量存储

        Args:
            keywords (List[str]): 关键词列表

        Returns:
            List[Document]: 文档列表
        """
        raise NotImplementedError
    
    def get_cites_from_answer(self, answer: str, doc_list: List[Document]) -> Tuple[Dict[int, int], List[Document]]:
        """从回答中获取引用信息，并且合并同一个文档的引用

        Args:
            answer (str): 回答文本
            doc_list (List[Document]): 文档列表

        Returns:
            Tuple[Dict[int, int], List[Document]]: 引用映射和使用的文档列表
        """
        cites_list = []
        match = re.findall(r'【(\d+)†source】', answer)
        if match:
            cites_list = sorted(set([int(i) for i in match]))
            
        used_doc_list = []
        doc_to_new_index = {}
        
        for ref in cites_list:
            # 检查索引是否有效
            if ref > len(doc_list):
                continue
                
            doc = doc_list[ref-1]
            if doc.filename not in doc_to_new_index:
                doc_to_new_index[doc.filename] = len(used_doc_list) + 1
                used_doc_list.append(doc)
                
        ref_mapping = {ref: doc_to_new_index[doc_list[ref-1].filename] 
                      for ref in cites_list 
                      if (ref - 1) < len(doc_list) and doc_list[ref-1].filename in doc_to_new_index}
                      
        return ref_mapping, used_doc_list 