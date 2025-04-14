from core.rags.base import BaseRag
from core.config import config
from utils.log import logger

class QuestionRag(BaseRag):
    """通用问答场景下的召回"""

    query_template = """
请分析用户的问题，从以下三个维度生成用于向量数据库检索的查询词：
1. 核心问题：提取最核心的问题
2. 背景知识：提取问题相关的背景知识
3. 简略解答：提取问题相关的简略解答

每个查询单独一行，只需要返回查询词，不需要任何解释。
用户问题: {query}
"""

    async def search_for_docs(self) -> list:
        """获取查询文档"""
        query = self.query_template.format(query=self.query)
        _, resp = await self.llm.chat(prompt=query)
        
        # 提取所有查询词
        query_list = [line.strip() for line in resp.split("\n") if line.strip()]
        query_list.append(self.query)
        logger.info(f"查询：{query_list}")

        return await self.search_vector(query_list, config.qa.threshold)