from fastapi import APIRouter, Depends
from pydantic import BaseModel
import json
from utils.log import logger
from fastapi.responses import StreamingResponse
from .utils import *
from core.vector.base import VectorStoreBase
from core.embeddings.base import EmbeddingAgent
from core.llms.base import AsyncBaseChatCOTModel
from core.rags.law import LawRag
from core.ranks import AsyncRankAgent
from fastapi import UploadFile, File
import httpx
from core.config import config
from datetime import datetime

law_router = APIRouter(prefix="/law")

async def generate_analysis(msg: str, history: list, use_cot_model: bool, system_prompt: str, query_templat: str = "", llm: AsyncBaseChatCOTModel = None, cot_llm: AsyncBaseChatCOTModel = None):
    """通用分析生成函数"""
    prompt = query_templat.format(msg=msg)
    
    history.insert(0, {'role': "system", 'content': system_prompt}) # 插入系统提示词
    history.append({'role': "user", 'content': prompt}) # 插入用户输入

    all_thinking = ""
    all_answer = ""
    if use_cot_model:
        async for thinking, resp in await cot_llm.chat(messages=history, stream=True):
            all_thinking += thinking
            all_answer += resp
            yield {"thinking": all_thinking, "answer": all_answer}
    else:
        async for _, resp in await llm.chat(messages=history, stream=True):
            all_answer += resp
            yield {"answer": all_answer}

    history.pop(0) # 移除系统提示词
    history.pop() # 移除用户输入
    history.append({'role': "user", 'content': msg})
    history.append({'role': "assistant", 'content': all_answer})
    result = {"answer": "<end>", "history": history}
    yield result

class lawqa(BaseModel):
    msg: str
    history: list = []
    collection_name: str | None = None
    use_cot_model: bool = False

@law_router.post("/lawqa")
async def post_lwa_qa_endpoint(input:lawqa, milvus: VectorStoreBase = Depends(get_milvus_store), text_embedder:EmbeddingAgent = Depends(get_embedding), llm:AsyncBaseChatCOTModel = Depends(get_llm),cot_llm:AsyncBaseChatCOTModel = Depends(get_llm_cot), reranker:AsyncRankAgent = Depends(get_reranker)):
    """法律相关问答"""
    async def generate():
        query = input.msg
        history = input.history
        collection_name = input.collection_name
        use_cot_model = input.use_cot_model

        rag = LawRag(query=query,collection_name=collection_name,department=None,messages=history,text_embedder=text_embedder,vector_store=milvus,llm=llm, reranker=reranker)
        docs = await rag.choose_doc_for_answer()

        doc_str = ""
        doc_list = []
        for doc in docs:
            doc_str += f'法条名称：\n{doc.filename}\n法条内容:\n{doc.text}\n'
            doc_list.append({"doc_name":doc.filename,"doc_content":doc.text})
        if doc_str == "":
            doc_str = "没有相关知识"

        query_templat = doc_str + "用户案件描述:\n{msg}"
        system_str = "你是基层的智能调解员，你会请根据用户的案件描结合相关的法律，站在调解员的角度给出建议。如果不涉及纠纷或者法律问题，请直接回答用户的问题。"
        
        async for data in generate_analysis(
            msg=query,
            history=history,
            use_cot_model=use_cot_model,
            system_prompt=system_str,
            query_templat=query_templat,
            llm=llm,
            cot_llm=cot_llm
        ):
            if data["answer"] == "<end>":
                data["doc_list"] = doc_list
                yield f"data: {json.dumps(data)}\n\n"
            else:
                yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(generate(),media_type="text/event-stream")

@law_router.post("/conflict_warning")
async def conflict_warning_endpoint(input: lawqa, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    """矛盾激化为极端案事件预警模型"""
    system_prompt = "你是一个分析矛盾纠纷是否可能激化为极端案事件的专家。请分析提供的矛盾纠纷数据，预测它激化为极端案事件的概率（以百分比表示）。分析时要考虑矛盾的性质、涉事人员状态、历史纠纷情况等因素。输出格式应为：\n1. 结果：当前矛盾激化为极端案事件可能性为X%，简单总结分析结果。\n2. 推理过程：详细分析你是如何得出这个结论的。"
    query_templat = "分析以下矛盾纠纷数据，预测它激化为极端案事件的概率：\n{msg}"
    
    async def generate():
        async for data in generate_analysis(
            msg=input.msg, 
            history=input.history, 
            use_cot_model=input.use_cot_model, 
            system_prompt=system_prompt,
            query_templat=query_templat,
            llm=llm,
            cot_llm=cot_llm
        ):
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(generate(),media_type="text/event-stream")

@law_router.post("/key_person_warning")
async def key_person_warning_endpoint(input: lawqa, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    """重点人风险预警模型"""
    system_prompt = "你是一个分析重点人风险的专家。请分析提供的重点人相关数据，预测其发生极端案事件的概率（以百分比表示）。分析时要考虑人员背景、历史行为、近期状态等因素。输出格式应为：\n1. 结果：当前重点人发生极端案事件风险概率为X%，简单总结分析结果。\n2. 推理过程：详细分析你是如何得出这个结论的。"
    query_templat = "分析以下重点人相关数据，预测其发生极端案事件的概率：\n{msg}"
    
    async def generate():
        async for data in generate_analysis(
            msg=input.msg, 
            history=input.history, 
            use_cot_model=input.use_cot_model, 
            system_prompt=system_prompt,
            query_templat=query_templat,
            llm=llm,
            cot_llm=cot_llm
        ):
            yield f"data: {json.dumps(data)}\n\n"
    
    return StreamingResponse(generate(),media_type="text/event-stream")

@law_router.post("/analysis_report")
async def analysis_report_endpoint(input: lawqa, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    """分析报告生成模型"""
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = "你是一个专业的矛盾数据分析师，负责生成区域数据分析报告。请根据用户提供的矛盾纠纷数据，生成全面的分析报告。分析应包括时间趋势、区域分布、类型等维度，对现状及治理结果进行解读总结，挖掘工作薄弱环节，并给出市级政法委工作建议。"
    query_template = f"现在的日期是{now_time}，请注意报告生成的日期。\n"+"{msg}"

    async def generate():
        async for data in generate_analysis(
            msg=input.msg, 
            history=input.history, 
            use_cot_model=input.use_cot_model, 
            system_prompt=system_prompt,
            query_templat=query_template,
            llm=llm,
            cot_llm=cot_llm
        ):
            yield f"data: {json.dumps(data)}\n\n"
    
    return StreamingResponse(generate(),media_type="text/event-stream")

@law_router.post("/asr")
async def asr_endpoint(file: UploadFile = File(...)):
    """语音识别"""
    async with httpx.AsyncClient() as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        response = await client.post(config.asr.url, files=files)
        return response.json()
