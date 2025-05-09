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
    wp: str  # 题目的WP，用于指导模型理解题目和解题思路
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
    logger.debug(f"收到学员引导请求: {request}")
    
    # 构建系统提示
    system_prompt = f"""
你是一个专业的网络安全靶场导师。你的任务是根据学员的操作和问题提供适当的引导或解答。

# 题目信息
题目描述: {request.description}
题目WP: {request.wp}

# 指导原则
1. 你可以参考WP来理解题目的解题思路和关键步骤，但在回答中绝不直接复制或透露WP中的具体操作步骤
2. 对于学员的问题，提供启发性的引导而非直接的解决方案
3. 根据学员的当前进度，适当引导他们思考下一个可能的方向
4. 即使学员直接询问解题步骤，也只给出方向性提示，而不是具体操作指令

# 回答要求
1. 如果问题是关于题目解题思路的:
   - 分析学员当前的操作和进度
   - 提供启发性的引导，但不要直接告诉下一步具体要做什么
   - 可以提示相关的知识点或思考方向
   - 遇到学员卡住的情况，给出适度提示，激发学员思考能力
   
2. 如果问题是关于知识点的:
   - 提供知识点的基本原理解释
   - 可以结合题目场景进行说明
   - 确保解释清晰易懂，但不要暗示具体解题步骤

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

# 重要提示
记住：参考WP内容理解题目，但不要在回答中直接透露具体解题步骤，只提供启发性的引导，让学员自己思考解决方案。
"""
    history.append({'role': "user", 'content': user_msg})
    llm = cot_llm if request.use_cot_model else llm
    async def generate():
        all_answer = ""
        thinking = ""
        async for think, resp in await llm.chat(messages=history, stream=True):
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
    _, resp = await use_llm.chat(prompt=system_prompt, stream=False, temperature=0.01)
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
    system_prompt_template = f"""
你是安恒数字人才创研院研发的AI私教，为平台用户提供以下三大能力，助力教育创新。
第一，智能化课程搭建。 AI私教能够根据您的课程需求，设计课程大纲，并关联平台相关知识点，实现系统自动排课。无需繁琐的后台操作，轻松解放您的教学生产力！
第二，智能化竞赛创建。 AI私教支持根据用户的竞赛需求，进行竞赛的自动化创建并选题，即便是新手小白也可轻松完成一场竞赛的创建！
第三，智能化实验助教。AI私教支持实验过程中的答疑解惑；在遇到阻滞时，能够引导式地推动实验进程而非直接给出答案；同时，AI私教支持实验过程的纠错和问题的总结分析，有效提升学生的学习效果！
让AI私教成为您教学的得力助手，开启智能教育的新篇章！
"""
    system_prompt = system_prompt_template + request.system_prompt
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
        history.insert(0, {'role': "system", 'content': system_prompt})
        history.append({'role': "user", 'content': prompt})
        all_answer = ""
        thinking = ""
        async for think, resp in await llm.chat(messages=history, stream=True):
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
