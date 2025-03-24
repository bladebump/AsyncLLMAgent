from .base import EmbeddingAgent
from typing import List, Union, Dict, Any
import aiohttp
import json
from utils.log import logger

class SiliconEmbeddingAgent(EmbeddingAgent):
    """
    通过硅基流动API获取嵌入向量的代理
    """
    
    def __init__(self, url: str, api_key: str, model: str = "Pro/BAAI/bge-m3"):
        """
        初始化硅基流动嵌入代理
        
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

    async def get_embedding(self, query: str, task_type: str = "retrieval.passage") -> List[float]:
        """
        通过硅基流动API获取单个文本的嵌入向量
        
        Args:
            query: 需要获取嵌入的文本
            task_type: 任务类型
            
        Returns:
            嵌入向量，浮点数列表
        """
        payload = {
            "model": self.model,
            "input": query,
            "encoding_format": "float"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload, headers=self.headers) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(f"地址: {self.url},请求失败，状态码: {response.status}, 响应内容: {response_text}")
                    raise Exception(f"请求失败，状态码: {response.status}")
                
                response_json = await response.json()
                # 从硅基流动API的响应中提取embedding
                return response_json["data"][0]["embedding"]
    
    async def encode(self, query: Union[str, List[str]], task_type: str = "retrieval.passage") -> Union[List[float], List[List[float]]]:
        """
        获取一个或多个文本的嵌入向量
        
        Args:
            query: 单个文本或文本列表
            task_type: 任务类型
            
        Returns:
            单个嵌入向量或嵌入向量列表
        """
        if isinstance(query, str):
            return await self.get_embedding(query, task_type)
        else:
            # 如果是文本列表，需要分别获取每个文本的embedding
            embeddings = []
            for text in query:
                embedding = await self.get_embedding(text, task_type)
                embeddings.append(embedding)
            return embeddings 
        
