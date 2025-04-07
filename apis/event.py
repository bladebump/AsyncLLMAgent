from fastapi import APIRouter, Depends
from pydantic import BaseModel
import json
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, get_llm_cot, parse_markdown_json
import base64

event_router = APIRouter(prefix="/event")

class Frame(BaseModel):
    timestamp: int
    data: str # base64编码的内容

class EventPost(BaseModel):
    frame_list: list[Frame]
    use_cot_model: bool = False

@event_router.post("/event_analysis")
async def event_analysis(events: EventPost, llm:AsyncBaseChatCOTModel = Depends(get_llm),cot_llm:AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    frame_list = []
    for frame in events.frame_list:
        frame_list.append({
            "session_id": frame.session_id,
            "timestamp": frame.timestamp,
            "data": base64.b64decode(frame.data).decode("utf-8")
        })
    prompt = f"""
# 事件内容
{frame_list}

# 指令
以上是捕获到的tty log，请分析出用户做了什么操作,输出格式如下，使用json格式

# 输出格式
[{{
    "seesion_id": "会话id",
    "event_name": "用户做了什么操作",
    "event_input": "用户操作",
    "event_output": "用户操作的输出，输出比较短的时候，给出完整输出，输出比较长的时候，给出输出总结",
    "event_info": "操作的详细信息",
    "event_special": "操作的特殊信息",
    "event_start": "操作开始的时间戳",
    "event_end": "操作结束的时间戳"
}}]
"""
    use_llm = cot_llm if events.use_cot_model else llm
    _, resp = await use_llm.chat(prompt=prompt, stream=False)
    
    try:
        return {"code": 200, "error": "", "data": parse_markdown_json(resp)}
    except json.JSONDecodeError:
        return {"code": 500, "error": "json解析失败", "data": []}
