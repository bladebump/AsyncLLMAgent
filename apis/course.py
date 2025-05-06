from fastapi import APIRouter, Depends
from pydantic import BaseModel
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, get_llm_cot
from utils.log import logger
from apis.course_utils.schema import Course, CourseBaseInfo
from apis.course_utils.check import analyze_course_completeness, process_user_input, get_tags
from fastapi.responses import StreamingResponse
from core.schema import Message
import json

course_router = APIRouter(prefix="/course")

class CreateCourseRequest(BaseModel):
    course: str
    user_input: str
    history: list[dict]
    use_cot_model: bool = False
    token: str

@course_router.post("/create_course")
async def create_course(createCourseRequest: CreateCourseRequest, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    system_prompt = "你是一个课程创建助手，需要根据用户当前的课程配置情况和对话历史，引导用户填写剩余的课程信息，并确保课程配置的完整性。"
    llm = cot_llm if createCourseRequest.use_cot_model else llm
    
    course_json = createCourseRequest.course
    user_input = createCourseRequest.user_input
    history = createCourseRequest.history

    course_data = json.loads(course_json) if course_json else {}    
    if not course_data:
        course = Course(baseInfo=CourseBaseInfo())
    else:
        course = Course.model_validate(course_data)
    is_completed = False
    course, update_message = await process_user_input(course, user_input, history, llm, createCourseRequest.token)
    next_step, missing_fields = await analyze_course_completeness(course, user_input, llm)
    logger.debug(f"next_step: {next_step}")
    tags = await get_tags(createCourseRequest.token)
    
    if "课程配置完成" in next_step:
        is_completed = True
        prompt = f"""
课程配置已经完成，可以提交创建。以下是您的课程配置信息：

# 课程信息
{course.model_dump()}

向用户展示信息即可，不需要用户确认。
"""
    else:
        prompt = f"""
当前课程配置状态:
{course.model_dump()}

用户最新输入:
{user_input}

更新信息:
{update_message}

下一步需要询问的内容:
{next_step}

未填写的字段:
{missing_fields}

当前课程标签:
{tags}

请生成一个友好的响应，引导用户完成课程创建过程。响应应当:
1. 确认已经填写/更新的内容
2. 清晰指出下一步需要填写什么内容
3. 如果需要，提供填写示例或选项
4. 使用友好的对话语气
5. 在用户没有输入名称前，不要提示用户输入标签相关信息
"""
    # 将Course对象转换为JSON
    course_json = course.model_dump_json()
    async def generate():
        history.insert(0, Message.system_message(system_prompt))
        history.append(Message.user_message(prompt))
        all_answer = ""
        thinking = ""
        async for chunk_thinking, chunk_response in await llm.chat(messages=history, stream=True, temperature=0.01):
            thinking += chunk_thinking
            all_answer += chunk_response
            if createCourseRequest.use_cot_model:
                yield f"data: {json.dumps({'answer': all_answer, 'thinking': thinking})}\n\n"
            else:
                yield f"data: {json.dumps({'answer': all_answer})}\n\n"
        history.pop(0)
        history.pop()
        history.append({
            "role": "user",
            "content": user_input
        })
        history.append({
            "role": "assistant", 
            "content": all_answer
        })
        result = {
            "answer": "<end>",
            "is_completed": is_completed,
            "course": course_json,
            "history": history
        }
        yield f"data: {json.dumps(result)}\n\n"
        logger.debug(f"创建完成: {result}")
    return StreamingResponse(generate(), media_type="text/event-stream")