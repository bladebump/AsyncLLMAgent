from .base import AsyncRankAgent
from typing import List, Dict, Any
import aiohttp
import json
from utils.log import logger

class HttpRankAgent(AsyncRankAgent):
    """
    通过HTTP请求实现的重排序代理
    """
    
    def __init__(self, url: str, model_name: str = 'bce-reranker-base_v1'):
        """
        初始化HTTP重排序代理
        
        Args:
            url: 重排序服务的URL地址
            model_name: 使用的模型名称
        """
        self.url = url
        self.model_name = model_name

    async def rerank(self, query: str, passages: List[str]) -> Dict[str, Any]:
        """
        通过HTTP请求对检索到的文本段落进行重排序
        
        Args:
            query: 查询文本
            passages: 待重排序的文本段落列表
            
        Returns:
            Dict[str, Any]: 包含重排序结果的字典，通常包含以下键：
                - rerank_passages: 重排序后的文本段落
                - rerank_scores: 对应的分数
                - rerank_ids: 原始顺序的索引
        """
        data = {
            "query": query,
            "passages": passages,
        }
        
        default_result = {
            'rerank_passages': passages,
            'rerank_scores': [0 for _ in range(len(passages))],
            'rerank_ids': list(range(len(passages)))
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=data) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(f"地址: {self.url},请求失败，状态码: {response.status}, 响应内容: {response_text}")
                        return default_result
                    
                    response_json = await response.json()
                    return response_json
        except Exception as e:
            logger.error(f"重排序请求异常: {str(e)}")
            return default_result 