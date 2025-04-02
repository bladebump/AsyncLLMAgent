from fastapi import APIRouter, Depends
from pydantic import BaseModel
import json
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, get_llm_cot, parse_markdown_json

event_router = APIRouter(prefix="/event")

class Event(BaseModel):
    event_content: str
    use_cot_model: bool = False

@event_router.post("/event_analysis")
async def event_analysis(event: Event, llm:AsyncBaseChatCOTModel = Depends(get_llm),cot_llm:AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    prompt = f"""
# 事件内容
{event.event_content}

# 指令
以上是捕获到的tty log，请分析出用户做了什么操作,输出格式如下，使用json格式

# 输出格式
[{{
    "event_input": "用户操作",
    "event_name": "用户做了什么操作",
    "event_output": "用户操作的输出，输出比较短的时候，给出完整输出，输出比较长的时候，给出输出总结",
    "event_info": "操作的详细信息",
    "event_time": "操作的时间",
    "event_special_info": "操作的特殊信息"
}}]
"""
    use_llm = cot_llm if event.use_cot_model else llm
    _, resp = await use_llm.chat(prompt=prompt, stream=False)
    
    try:
        return parse_markdown_json(resp)
    except json.JSONDecodeError:
        return {"error": "解析失败", "raw_resp": resp}
