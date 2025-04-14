from typing import List, Dict, Any
from abc import ABC, abstractmethod

class AsyncRankAgent(ABC):
    """
    重排序代理的基类，定义了重排序的接口
    """
    
    @abstractmethod
    async def rerank(self, query: str, passages: List[str], top_n: int = 5) -> Dict[str, Any]:
        """
        对检索到的文本段落进行重排序
        
        Args:
            query: 查询文本
            passages: 待重排序的文本段落列表
            
        Returns:
            Dict[str, Any]: 包含重排序结果的字典，通常包含以下键：
                - rerank_passages: 重排序后的文本段落
                - rerank_scores: 对应的分数
                - rerank_ids: 原始顺序的索引
        """
        pass 