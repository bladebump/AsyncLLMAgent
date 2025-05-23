from fastapi import APIRouter, Depends
from pydantic import BaseModel
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, get_llm_cot, parse_markdown_json, parse_markdown_yaml
from events import ParserFactory, ParserType, Frame, Event
from utils.log import logger

event_router = APIRouter(prefix="/event")

class EventPost(BaseModel):
    frame_list: list[Frame]
    use_cot_model: bool = False
    request_id: str
    parser_type: ParserType = ParserType.MERGE

@event_router.post("/event_analysis")
async def event_analysis(events: EventPost, llm:AsyncBaseChatCOTModel = Depends(get_llm),cot_llm:AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    """事件分析，将frame转换成event"""
    logger.debug(f"frame_list: {events.frame_list}")
    parser = ParserFactory.create_parser(events.parser_type, events.frame_list)
    frame_list = parser.parse()
    logger.debug(f"Parse frame_list: {frame_list}")
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
   - event 之间不能有时间重叠，必须按时间顺序排列
   - 用户输入通常以回车键结束，但要注意特殊场景（如 vim 编辑模式）
   - 对于长输出，可以适当总结而不是完整展示
   - 如果命令失败了，请在输出中总结出有效信息，包括系统的关键提示，或者系统的建议
   - 将相关的连续操作合并为一个事件，例如：命令输入+Tab补全+回车执行应合并为一个完整事件
   - 命令输入可能包含特殊按键，如Tab补全(\\t)、方向键等，应将这些视为单个命令的一部分

# 输出格式
请以 YAML 格式输出分析结果，每个 event 包含以下字段：
```yaml
- event_name: "操作的简短描述"
  event_input: "用户输入的命令或操作"
  event_output: "系统响应（短输出完整展示，长输出总结）"
  event_info: "操作的详细说明"
  event_special: "需要特别说明的信息（如特殊模式、异常情况等）"
  event_start: "操作开始的时间戳"
  event_end: "操作结束的时间戳"
```

如果没有检测到任何事件，则输出：
无

# 注意事项
1. 确保时间戳的准确性，event_start 和 event_end 必须准确反映操作的时间范围
2. 对于复杂的操作序列，需要仔细分析输入和输出的对应关系
3. 注意识别特殊场景，如交互式程序、编辑模式等
4. 输出的值使用中文
5. 如果存在event，则输出格式必须严格符合YAML格式，不要包含其他内容，不存在event则输出无
6. 输出中不应该显示控制字符（如\\r\\n），应清理这些字符或替换为适当的描述
7. 对于Tab补全等操作，应将其视为命令输入的一部分，与后续执行合并为一个事件
"""
    use_llm = cot_llm if events.use_cot_model else llm
    _, resp, _ = await use_llm.chat(prompt=prompt, stream=False, temperature=0.01)
    if resp == "无":
        return {"code": 200, "error": "", "data": {"event_list": [], "request_id": events.request_id}}
    try:
        result = parse_markdown_yaml(resp)
        if isinstance(result, str):
            return {"code": 200, "error": "", "data": {"event_list": [], "request_id": events.request_id}}
        result = [Event(**item).model_dump() for item in result]
        return {"code": 200, "error": "", "data": {"event_list": result, "request_id": events.request_id}}
    except Exception as e:
        return {"code": 500, "error": f"YAML解析失败\n{resp}\n错误信息: {str(e)}", "data": {"event_list": [], "request_id": events.request_id}}

