from core.rags.base import BaseRag
from core.config import config

class QuestionRag(BaseRag):
    """通用问答场景下的召回"""

    query_template = """
请分析用户的问题，从以下三个维度提取关键词用于向量数据库检索：
1. 核心关键词：提取最相关的三个关键词
2. 背景知识：提取三个更基础或更广泛的相关概念
3. 详细解答：提取三个更深入或更具体的关键词

每个维度的关键词单独一行，用"维度："作为前缀。
用户问题: {query}
"""

    async def search_for_docs(self) -> list:
        """获取查询文档"""
        query = self.query_template.format(query=self.query)
        _, resp = await self.llm.chat(prompt=query)
        
        # 提取所有查询词
        query_list = [line.split("：")[1].strip() for line in resp.split("\n") if "：" in line]
        query_list.append(self.query)
        
        return await self.search_vector(query_list, config.qa.threshold)