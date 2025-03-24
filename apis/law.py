from fastapi import APIRouter, Depends
from pydantic import BaseModel
import json
import random
from utils.log import logger
from fastapi.responses import StreamingResponse
from .utils import *
from core.vector.base import VectorStoreBase
from core.embeddings.base import EmbeddingAgent
from core.llms.base import AsyncBaseLLMModel
from core.rags.law import LawRag
import os
import aiofiles

law_router = APIRouter(prefix="/law")

class lawqa(BaseModel):
    msg: str
    history: list = []
    collection_name: str | None = None

@law_router.post("/lawqa")
async def post_lwa_qa_endpoint(input:lawqa, milvus: VectorStoreBase = Depends(get_milvus_store), text_embedder:EmbeddingAgent = Depends(get_embedding), deepseek_llm:AsyncBaseLLMModel = Depends(get_llm)):
    """法律相关问答"""
    async def generate():
        query = input.msg
        history = input.history
        collection_name = input.collection_name
        system_str = ""

        expert_contact_information = {}

        file_path = os.path.dirname(__file__)
        expert_path = os.path.join(file_path,"expert.json")
        async with aiofiles.open(expert_path, 'r', encoding='utf-8') as f:
            expert_contact_information = json.loads(await f.read())

        rag = LawRag(query=query,collection_name=collection_name,department=None,messages=history,text_embedder=text_embedder,milvus_client=milvus,llm=deepseek_llm)
        docs = await rag.choose_doc_for_answer()

        doc_str = ""
        doc_list = []
        for doc in docs:
            doc_str += f'法条名称：\n{doc.filename}\n法条内容:\n{doc.text}\n'
            doc_list.append({"doc_name":doc.filename,"doc_content":doc.text})
        if doc_str == "":
            doc_str = "没有相关知识"

        query_template = doc_str + f"用户案件描述:\n{query}"
        if system_str == "":
            system_str = "你是基层的智能调解员，你会请根据用户的案件描结合相关的法律，站在调解员的角度给出建议。"
        if len(history) == 0:
            history.append({'role':"system", 'content':system_str})
        history.append({'role':"user", 'content':query_template})

        async for resp in deepseek_llm.chat(messages=history,stream=True):
            data = json.dumps({"answer":resp})
            yield f"data: {data}\n\n"

        old_resp = resp
        expert_type = random.choice(['人民调解员','网格员'])
        expert = random.choice(expert_contact_information[expert_type])
        old_resp += f"\n基于以上情况，建议联系\n{expert_type}-{expert['姓名']},联系方式为{expert['联系方式']}"
        expert_type = random.choice(['民警','辅警'])
        expert = random.choice(expert_contact_information[expert_type])
        old_resp += f"\n{expert_type}-{expert['姓名']},联系方式为{expert['联系方式']}"
        expert_type = random.choice(['律师','心理咨询师'])
        expert = random.choice(expert_contact_information[expert_type])
        old_resp += f"\n{expert['单位']}-{expert_type}-{expert['姓名']},联系方式为{expert['联系方式']}"
        data = json.dumps({"answer":old_resp})
        yield f"data: {data}\n\n"
        resp = old_resp
        history.pop()
        history.append({'role':"user", 'content':query})
        history.append({'role':"assistant", 'content':resp})
        result = {"answer":"<end>","history":history,"doc_list":doc_list}
        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(generate(),media_type="text/event-stream")