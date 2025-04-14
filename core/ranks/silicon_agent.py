from .base import AsyncRankAgent
from typing import List, Dict, Any
import aiohttp
import json
from utils.log import logger

class SiliconRankAgent(AsyncRankAgent):
    """
    通过硅基流动API进行重排序的代理
    """
    
    def __init__(self, url: str, api_key: str, model: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化硅基流动重排序代理
        
        Args:
            url: 硅基流动API的URL地址
            api_key: API密钥
            model: 使用的模型名称
        """
        self.url = url
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def rerank(self, query: str, passages: List[str], top_n: int = 5) -> Dict[str, Any]:
        """
        通过硅基流动API对文档进行重排序
        
        Args:
            query: 查询文本
            passages: 待重排序的文档列表
            top_n: 返回前N个结果
            
        Returns:
            包含重排序结果的字典，包括重排序后的文档、分数和ID
        """
        payload = {
            "model": self.model,
            "query": query,
            "documents": passages,
            "top_n": top_n,
            "return_documents": False,
            "max_chunks_per_doc": 1024,
            "overlap_tokens": 80
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload, headers=self.headers) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(f"地址: {self.url},请求失败，状态码: {response.status}, 响应内容: {response_text}")
                    # 如果请求失败，返回原始文档列表，分数设为0
                    return {
                        "rerank_passages": passages,
                        "rerank_scores": [0.0] * len(passages),
                        "rerank_ids": list(range(len(passages)))
                    }
                
                response_json = await response.json()
                results = response_json["results"]
                
                # 提取重排序结果
                reranked_passages = []
                scores = []
                ids = []
                
                for result in results:
                    reranked_passages.append(passages[result["index"]])
                    scores.append(result["relevance_score"])
                    ids.append(result["index"])
                
                return {
                    "rerank_passages": reranked_passages,
                    "rerank_scores": scores,
                    "rerank_ids": ids
                } 