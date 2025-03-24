from typing import List, Union
from abc import ABC, abstractmethod

class EmbeddingAgent(ABC):
    """
    嵌入代理的基类，定义了获取嵌入的接口
    """
    
    @abstractmethod
    async def get_embedding(self, query: str, task_type: str = "retrieval.passage") -> List[float]:
        """
        获取单个文本的嵌入向量
        
        Args:
            query: 需要获取嵌入的文本
            task_type: 任务类型
            
        Returns:
            嵌入向量，浮点数列表
        """
        pass
    
    @abstractmethod
    async def encode(self, query: Union[str, List[str]], task_type: str = "retrieval.passage") -> Union[List[float], List[List[float]]]:
        """
        获取一个或多个文本的嵌入向量
        
        Args:
            query: 单个文本或文本列表
            task_type: 任务类型
            
        Returns:
            单个嵌入向量或嵌入向量列表
        """
        pass 