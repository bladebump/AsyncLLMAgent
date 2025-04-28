from fastapi import APIRouter, Depends
from pydantic import BaseModel
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, get_llm_cot
from utils.log import logger
from apis.course_utils.schema import Course, CourseBaseInfo
from apis.course_utils.check import analyze_course_completeness, process_user_input
from fastapi.responses import StreamingResponse
from core.schema import Message
import json

course_router = APIRouter(prefix="/course")

class CreateCourseRequest(BaseModel):
    course: str
    user_input: str
    history: list[dict]
    use_cot_model: bool = False

@course_router.post("/create_course")
async def create_course(createCourseRequest: CreateCourseRequest, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    logger.debug(f"收到创建课程请求: {createCourseRequest}")
    system_prompt = "你是一个课程创建助手，需要根据用户当前的课程配置情况和对话历史，引导用户填写剩余的课程信息，并确保课程配置的完整性。"
    llm = cot_llm if createCourseRequest.use_cot_model else llm
    
    course_json = createCourseRequest.course
    # print(f"course_json: {course_json}")        
    user_input = createCourseRequest.user_input
    # print(f"user_input: {user_input}")
    history = createCourseRequest.history
    # print(f"history: {history}")

    course_data = json.loads(course_json) if course_json else {}    
    # print(f"course_data: {course_data}")
    if not course_data:
        course = Course(baseInfo=CourseBaseInfo())
    else:
        course = Course.model_validate(course_data)
    print(f"course: {course}")
    is_completed = False
    # 处理用户输入并更新course对象
    course, update_message = await process_user_input(course, user_input, history, llm)
    print(f"course: {course}")
    # print(f"update_message: {update_message}")
    # 检查课程配置的完整性，确定下一步需要填写的信息
    next_step, missing_fields = await analyze_course_completeness(course, user_input, llm)
    # print(f"next_step: {next_step}")
    # print(f"missing_fields: {missing_fields}")

    if next_step == "课程配置完成":
        is_completed = True
        prompt = f"""
课程配置已经完成，可以提交创建。以下是您的课程配置信息：

# 课程信息
{course.model_dump()}

请确认以上信息无误，如需修改可以告诉我，或者直接提交创建。
"""
    else:
        prompt = f"""
当前课程配置状态:
{course.model_dump_json()}

用户最新输入:
{user_input}

更新信息:
{update_message}

下一步需要询问的内容:
{next_step}

未填写的字段:
{missing_fields}

请生成一个友好的响应，引导用户完成课程创建过程。响应应当:
1. 确认已经填写/更新的内容
2. 清晰指出下一步需要填写什么内容
3. 如果需要，提供填写示例或选项
4. 使用友好的对话语气
"""
    # 将Course对象转换为JSON
    course_json = course.model_dump_json()
    # print(f"course_json: {course_json}")
    async def generate():
        history.insert(0, Message.system_message(system_prompt))
        history.append(Message.user_message(prompt))
        # print(f"history: {history}")
        all_answer = ""
        thinking = ""
        async for chunk_thinking, chunk_response in await llm.chat(messages=history, stream=True):
            # print(f"chunk_thinking: {chunk_thinking}")
            # print(f"chunk_response: {chunk_response}")
            thinking += chunk_thinking
            all_answer += chunk_response
            if createCourseRequest.use_cot_model:
                yield f"data: {json.dumps({'answer': all_answer, 'thinking': thinking})}\n\n"
            else:
                yield f"data: {json.dumps({'answer': all_answer})}\n\n"
        history.pop(0)
        history.pop()
        # print(f"history: {history}")
        # 更新历史记录
        history.append({
            "role": "user",
            "content": user_input
        })
        history.append({
            "role": "assistant", 
            "content": all_answer
        })
        # print(f"history: {history}")
        result = {
            "answer": "<end>",
            "is_completed": is_completed,
            "course": course_json,
            "history": history
        }
        # print(f"result: {result}")
        yield f"data: {json.dumps(result)}\n\n"
    # print("返回结果")
    return StreamingResponse(generate(), media_type="text/event-stream")