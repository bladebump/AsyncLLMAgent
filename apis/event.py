from fastapi import APIRouter, Depends
from pydantic import BaseModel
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, get_llm_cot, parse_markdown_json
import base64
from events.parse import FrameParser, Frame

event_router = APIRouter(prefix="/event")

class EventPost(BaseModel):
    frame_list: list[Frame]
    use_cot_model: bool = False

@event_router.post("/event_analysis")
async def event_analysis(events: EventPost, llm:AsyncBaseChatCOTModel = Depends(get_llm),cot_llm:AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    frame_list = FrameParser(events.frame_list).parse()
    prompt = f"""
# 任务说明
你是一个专业的终端操作分析专家。你的任务是从连续的终端日志中识别和分析用户的操作序列。

# 输入数据
以下是按时间顺序排列的终端日志片段：
{frame_list}

# 分析要求
1. 每个用户操作（event）应该包含：
   - 用户输入的命令或操作
   - 系统或程序的响应输出
   - 操作的时间范围（开始和结束时间戳）

2. 分析原则：
   - 一个 event 必须包含完整的用户操作和对应的系统响应
   - event 之间不能有时间重叠，必须按时间顺序排列
   - 用户输入通常以回车键结束，但要注意特殊场景（如 vim 编辑模式）
   - 对于长输出，可以适当总结而不是完整展示

# 输出格式
请以 JSON 数组格式输出分析结果，每个 event 包含以下字段：
[
    {{
        "event_name": "操作的简短描述",
        "event_input": "用户输入的命令或操作",
        "event_output": "系统响应（短输出完整展示，长输出总结）",
        "event_info": "操作的详细说明",
        "event_special": "需要特别说明的信息（如特殊模式、异常情况等）",
        "event_start": "操作开始的时间戳",
        "event_end": "操作结束的时间戳"
    }}
]

# 注意事项
1. 确保时间戳的准确性，event_start 和 event_end 必须准确反映操作的时间范围
2. 对于复杂的操作序列，需要仔细分析输入和输出的对应关系
3. 注意识别特殊场景，如交互式程序、编辑模式等
4. 输出的值使用中文
5. 事件必须和用户输入的命令或操作对应，没有相应的输入，则不输出event
6. 如果没有相应的event，则输出[]
7. 输出格式必须符合json格式，不要包含其他内容
"""
    use_llm = cot_llm if events.use_cot_model else llm
    _, resp = await use_llm.chat(prompt=prompt, stream=False, temperature=0.01)
    
    try:
        return {"code": 200, "error": "", "data": parse_markdown_json(resp)}
    except Exception as e:
        return {"code": 500, "error": f"json解析失败\n{resp}\n错误信息: {str(e)}", "data": []}
