from core.rags.base import BaseRag
from config import LAW_QA_THOULD

class LawRag(BaseRag):
    """问答场景下的召回"""

    query_template = """
这里有用户遇到的法律相关描述，请结合描述，输出应该查询哪方面的法律相关，方便去向量数据库检索。输出最相关的三个检索关键词，仅仅输出查询的关键词，一行一个。用户输入: {query}
"""

    async def search_for_docs(self) -> list:
        """获取查询文档"""
        query = self.query_template.format(query=self.query)
        resp = await self.llm.chat(prompt=query)
        query_list = resp.split("\n")

        query_list = [query for query in query_list if (query != "") and (len(query) >= 1)]
        query_list.append(self.query)
        
        return await self.search_vector(query_list,LAW_QA_THOULD)
