from fastapi import APIRouter, Depends
from pydantic import BaseModel
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, get_llm_cot
from events import Event
from utils.log import logger
from fastapi.responses import StreamingResponse
import json

guide_router = APIRouter(prefix="/guide")

class GuideRequest(BaseModel):
    wp: str  # 题目的WP
    description: str  # 题目描述
    events: list[Event]  # 学员操作event
    history: list[dict] = []  # 对话历史记录
    question: str  # 学员的问题

@guide_router.post("/student_guide")
async def student_guide(request: GuideRequest, llm: AsyncBaseChatCOTModel = Depends(get_llm)):
    """
    学员引导接口，根据题目信息、学员操作和问题提供引导或解答
    """
    logger.debug(f"收到学员引导请求: {request}")
    
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

    async def generate():
        all_answer = ""
        async for _, resp in await llm.chat(messages=history, stream=True):
            all_answer += resp
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
    logger.debug(f"收到对话总结请求: {request}")
    
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
    _, resp = await use_llm.chat(prompt=system_prompt, stream=False, temperature=0.01)
    return {
        "code": 200,
        "error": "",
        "data": {
            "summary": resp,
        }
    }