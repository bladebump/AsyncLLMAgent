from .base import EmbeddingAgent
from typing import List, Union
import aiohttp
import json
from utils.log import logger

class http_EmbeddingAgent(EmbeddingAgent):
    """
    通过HTTP请求获取嵌入向量的代理
    """
    
    def __init__(self, url: str, model_name: str = 'bce-embedding-base_v1'):
        """
        初始化HTTP嵌入代理
        
        Args:
            url: 嵌入服务的URL地址
            model_name: 使用的模型名称
        """
        self.url = url
        self.model_name = model_name

    async def get_embedding(self, query: str, task_type: str = "retrieval.passage") -> List[float]:
        """
        通过HTTP请求获取单个文本的嵌入向量
        
        Args:
            query: 需要获取嵌入的文本
            task_type: 任务类型
            
        Returns:
            嵌入向量，浮点数列表
        """
        data = {
            "texts": query,
            "task_type": task_type
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=data) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(f"地址: {self.url},请求失败，状态码: {response.status}, 响应内容: {response_text}")
                    raise Exception(f"请求失败，状态码: {response.status}")
                
                response_json = await response.json()
                return response_json["embeddings"]
    
    async def encode(self, query: Union[str, List[str]], task_type: str = "retrieval.passage") -> Union[List[float], List[List[float]]]:
        """
        获取一个或多个文本的嵌入向量
        
        Args:
            query: 单个文本或文本列表
            task_type: 任务类型
            
        Returns:
            单个嵌入向量或嵌入向量列表
        """
        result = await self.get_embedding(query, task_type)
        return result 