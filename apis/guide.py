from fastapi import APIRouter, Depends
from pydantic import BaseModel
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, get_llm_cot
from events import Event
from utils.log import logger
from fastapi.responses import StreamingResponse
import datetime
import json

guide_router = APIRouter(prefix="/guide")

class GuideRequest(BaseModel):
    wp: str  # 题目的WP
    description: str  # 题目描述
    events: list[Event]  # 学员操作event
    history: list[dict] = []  # 对话历史记录
    question: str  # 学员的问题
    use_cot_model: bool = False

@guide_router.post("/student_guide")
async def student_guide(request: GuideRequest, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    """
    学员引导接口，根据题目信息、学员操作和问题提供引导或解答
    """
    # 构建系统提示
    system_prompt = f"""
你是一个专业的网络安全靶场导师。你的任务是根据学员的操作和问题提供适当的引导或解答。

# 题目信息
题目描述: {request.description}
题目WP: {request.wp}

# 回答要求
1. 如果问题是关于题目解题思路的:
   - 分析学员当前的操作和进度
   - 提供适当的引导，但不要直接告诉下一步要做什么
   - 可以提示相关的知识点或方向
   
2. 如果问题是关于知识点的:
   - 提供详细的知识点解释
   - 可以结合题目场景进行说明
   - 确保解释清晰易懂

3. 回答格式:
   - 使用中文回答
   - 保持专业但友好的语气
   - 避免使用过于技术性的术语，除非必要
   - 回答要简洁明了，重点突出
"""
    
    # 初始化历史记录
    history = request.history
    if len(history) == 0:
        history.append({'role': "system", 'content': system_prompt})
    elif history[0]['role'] != "system":
        history.insert(0, {'role': "system", 'content': system_prompt})
    
    # 构建包含操作记录的用户消息
    user_msg = f"""
# 学员操作记录
{request.events}

# 学员问题
{request.question}
"""
    history.append({'role': "user", 'content': user_msg})
    llm = cot_llm if request.use_cot_model else llm
    async def generate():
        all_answer = ""
        thinking = ""
        async for think, resp, _ in await llm.chat(messages=history, stream=True):
            thinking += think
            all_answer += resp
            if request.use_cot_model:
                yield f"data: {json.dumps({'answer': all_answer, 'thinking': thinking})}\n\n"
            else:
                yield f"data: {json.dumps({'answer': all_answer})}\n\n"

        # pop system message
        history.pop(0)
        # 使得history为原始输入
        history.pop()
        history.append({'role': "user", 'content': request.question})
        history.append({'role': "assistant", 'content': all_answer})
        yield f"data: {json.dumps({'answer': '<end>', 'history': history})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream") 

class ConversationSummaryRequest(BaseModel):
    history: list[dict]  # 对话历史记录
    use_cot_model: bool = False  # 是否使用COT模型

@guide_router.post("/conversation_summary")
async def conversation_summary(request: ConversationSummaryRequest, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    """
    总结对话主题的接口
    """
    system_prompt = f"""
你是一个专业的对话主题分析助手。你的任务是从对话历史中提取出一个对话的主题。

请按照以下要求进行总结：
1. 识别对话中的主要主题
2. 提取出一个最能代表对话内容的主题
3. 主题需要简洁明了,不要超过10个字

输出格式要求：
- 使用中文
- 简明扼要
- 重点突出

对话历史：
{request.history}
"""
    use_llm = cot_llm if request.use_cot_model else llm
    _, resp, _ = await use_llm.chat(prompt=system_prompt, stream=False, temperature=0.01)
    return {
        "code": 200,
        "error": "",
        "data": {
            "summary": resp,
        }
    }

class UserGuideRequest(BaseModel):
    user_input: str  # 用户输入
    history: list[dict]  # 对话历史记录
    use_cot_model: bool = False  # 是否使用COT模型
    system_prompt: str = ""  # 系统提示

@guide_router.post("/user_guide")
async def user_guide(request: UserGuideRequest, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    """平台用户对话接口，解答用户的问题"""
    history = request.history
    llm = cot_llm if request.use_cot_model else llm
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt = f"""
你是一个专业的智能助手，能够回答用户的各种问题，提供准确、有用的信息。

在回答时，请注意以下几点：
- 当前时间：{cur_time}
- 根据用户问题提供最相关、最准确的回答。
- 对于列举类问题(如列举旅游景点、编程库等)，将答案控制在10个要点以内，优先提供最相关、信息最完整的选项。
- 如果回答很长，请结构化、分段落总结。如果需要分点作答，尽量控制在5个点以内，并合并相关内容。
- 对于客观类问答，如果答案非常简短，可以适当补充一到两句相关信息，丰富内容。
- 选择美观、易读的回答格式，确保可读性强。
- 保持与用户提问相同的语言回答。
- 对于编程相关问题，提供简洁、有效的代码示例和解释。
- 对于旅游、生活类问题，提供实用的建议和信息。
- 输出使用markdown格式。

请始终以礼貌、专业的态度回应用户，提供最有帮助的信息。

用户输入：
{request.user_input}
"""
    async def generate():
        history.insert(0, {'role': "system", 'content': request.system_prompt})
        history.append({'role': "user", 'content': prompt})
        all_answer = ""
        thinking = ""
        async for think, resp, _ in await llm.chat(messages=history, stream=True):
            thinking += think
            all_answer += resp
            if request.use_cot_model:
                yield f"data: {json.dumps({'answer': all_answer, 'thinking': thinking})}\n\n"
            else:
                yield f"data: {json.dumps({'answer': all_answer})}\n\n"
        history.pop(0)
        history.pop()
        history.append({'role': "user", 'content': request.user_input})
        history.append({'role': "assistant", 'content': all_answer})
        yield f"data: {json.dumps({'answer': '<end>', 'history': history})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream") 
