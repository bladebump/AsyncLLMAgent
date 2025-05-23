from fastapi import APIRouter, Depends
from pydantic import BaseModel
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, get_llm_cot
from utils.log import logger
import json
from apis.competition_utils.schema import Competition, CompetitionBaseInfo
from apis.competition_utils.check import analyze_competition_completeness, process_user_input
from fastapi.responses import StreamingResponse
from core.schema import Message
import time

competition_router = APIRouter(prefix="/competition")

class WPFormat(BaseModel):
    wp: str
    format_yaml: str
    use_cot_model: bool = False

@competition_router.post("/wp_format")
async def wp_format(wp_format: WPFormat, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    """格式化WriteUp"""
    logger.debug(f"收到WP格式化请求: {wp_format}")
    llm = cot_llm if wp_format.use_cot_model else llm
    prompt = f"""请将以下实验报告（writeup/wp）转换为结构化的YAML格式题目信息卡片。
我会提供一份网络安全相关的实验报告或解题步骤，请你仔细分析其中的信息，并按照下面的YAML模板提取关键要素：
```yaml
{wp_format.format_yaml}
```

请确保:
1. 每个步骤应该是完整的，如果下一步依赖上一步的结果，应该明确指出
2. 输入和预期结果要具体，便于复现
3. 关键观察要突出重点发现
4. 标签要精准反映题目特点和所需技能
5. 输出必须是严格的YAML格式，注意缩进和格式

以下是需要转换的实验报告:

{wp_format.wp}
"""
    thinking, response, _ = await llm.chat(prompt,stream=False)
    return {"code": 200, "error": None, "data": {"wp": response, "thinking": thinking}}

class CreateCompetitionRequest(BaseModel):
    competition: str  # JSON字符串
    user_input: str
    history: list[dict]
    use_cot_model: bool = False
    token: str = ""

@competition_router.post("/create_competition")
async def create_competition(createCompetitionRequest: CreateCompetitionRequest, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    """创建比赛"""
    logger.debug(f"收到创建竞赛请求: {createCompetitionRequest}")
    system_prompt = f"""你是一个竞赛创建助手，需要根据用户当前的竞赛配置情况和对话历史，引导用户填写剩余的竞赛信息。要创建的竞赛为网络安全相关竞赛，用来体现选手的网络安全攻防能力。
通常竞赛分为CTF、AWD、BTC，每种类型有不同的赛题设置和答题模式。
比赛创建完成不用说其他内容，只需要返回竞赛的详细信息。
"""
    llm = cot_llm if createCompetitionRequest.use_cot_model else llm
    
    # 从请求中提取信息
    competition_json = createCompetitionRequest.competition
    user_input = createCompetitionRequest.user_input
    history = createCompetitionRequest.history
    
    competition_data = json.loads(competition_json) if competition_json else {}    
    if not competition_data:
        competition = Competition()
    else:
        competition = Competition.model_validate(competition_data)
    is_completed = False
    # 处理用户输入并更新competition对象
    competition, update_message = await process_user_input(competition, user_input, history, llm, createCompetitionRequest.token)
    
    # 检查竞赛配置的完整性，确定下一步需要填写的信息
    next_step, missing_fields = await analyze_competition_completeness(competition, user_input, llm)
    logger.debug(f"竞赛配置完整性检查结果: {next_step}, {missing_fields}")
    if next_step == "竞赛配置完成":
        is_completed = True
        prompt = f"""
竞赛配置已经完成，可以提交创建。请将以下竞赛配置信息转换为用户友好的格式：

# 竞赛配置数据
{competition.model_dump()}

请对上述配置数据进行整理，以清晰、结构化的方式呈现关键信息，包括但不限于：
1. 竞赛名称、时间、简介等基本信息
2. 竞赛阶段和类型
3. 赛题设置和评分规则
4. 其他重要配置

注意：不要直接返回JSON格式，而是将数据转换为易于阅读的文本形式，使用适当的标题、分段和格式化。
"""
    else:
        prompt = f"""
当前竞赛配置状态:
{competition.model_dump()}

用户最新输入:
{user_input}

更新信息:
{update_message}

下一步需要询问的内容:
{next_step}

未填写的字段:
{missing_fields}

请生成一个友好的响应，引导用户完成竞赛创建过程。响应应当:
1. 确认已经填写/更新的内容
2. 清晰指出下一步需要填写什么内容
3. 如果需要，提供填写示例或选项
4. 使用友好的对话语气
5. 一个比赛可以有多个阶段，但是最少有一个阶段，请引导用户创建相应的阶段，如果有一个也可以继续添加。阶段可以有CTF（夺旗赛）、AWD（攻防赛）、BTC（闯关赛）。引导用户添加阶段的时候，仅引导用户输入阶段类型，比如CTF、AWD、BTC。
6. 在还有未填写字段的内容时候，请勿向用户介绍可以提交。
7. 给出适当的样例，但是不要建议用户使用默认配置。
"""
    # 将Competition对象转换为JSON
    competition_json = competition.model_dump_json()
    async def generate():
        history.insert(0, Message.system_message(system_prompt))
        history.append(Message.user_message(prompt))
        all_answer = ""
        thinking = ""
        async for chunk_thinking, chunk_response, _ in await llm.chat(messages=history, stream=True, temperature=0.01):
            thinking += chunk_thinking
            all_answer += chunk_response
            if createCompetitionRequest.use_cot_model:
                yield f"data: {json.dumps({'answer': all_answer, 'thinking': thinking})}\n\n"
            else:
                yield f"data: {json.dumps({'answer': all_answer})}\n\n"
        
        history.pop(0)
        history.pop()
        # 更新历史记录
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
            "competition": competition_json,
            "history": history
        }
        yield f"data: {json.dumps(result)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

