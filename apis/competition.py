from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, parse_markdown_json
from utils.log import logger
from core.schema import Message

competition_router = APIRouter(prefix="/competition")

# 竞赛相关数据模型
class CompetitionLevel(str):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class CompetitionTrack(str):
    SECURITY = "security"
    DEVELOPMENT = "development"
    OPERATION = "operation"
    INTEGRATION = "integration"

class CompetitionTeam(BaseModel):
    team_name: str = Field(..., description="团队名称")
    members: List[str] = Field(..., description="团队成员列表")

class Competition(BaseModel):
    name: str = Field(..., description="竞赛名称")
    description: str = Field(..., description="竞赛描述")
    start_date: date = Field(..., description="开始日期")
    end_date: date = Field(..., description="结束日期")
    max_participants: int = Field(..., description="最大参与人数")
    level: str = Field(..., description="难度级别", examples=[CompetitionLevel.EASY, CompetitionLevel.MEDIUM, CompetitionLevel.HARD])
    track: str = Field(..., description="竞赛赛道", examples=[CompetitionTrack.SECURITY, CompetitionTrack.DEVELOPMENT, CompetitionTrack.OPERATION, CompetitionTrack.INTEGRATION])
    teams: Optional[List[CompetitionTeam]] = Field(default=[], description="参赛团队")
    rules: str = Field(..., description="竞赛规则")
    prizes: Optional[Dict[str, str]] = Field(default={}, description="奖品设置")

class CompetitionCreate(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="对话消息历史")

class CompetitionResponse(BaseModel):
    code: int = Field(200, description="状态码")
    error: str = Field("", description="错误信息")
    data: Dict[str, Any] = Field(..., description="响应数据")

@competition_router.post("/create")
async def create_competition(request: CompetitionCreate, llm: AsyncBaseChatCOTModel = Depends(get_llm)):
    """
    通过对话方式创建竞赛，如果信息不完整，将引导用户提供缺失信息
    """
    # 将消息历史转换为Message对象
    history = [Message.from_history(msg) for msg in request.messages]
    
    # 构建系统提示
    system_prompt = """
    你是一个专业的竞赛创建助手。你的任务是通过对话引导用户创建一个完整的竞赛信息。
    
    你需要收集以下信息：
    1. 竞赛名称(name)
    2. 竞赛描述(description)
    3. 开始日期(start_date)，格式为YYYY-MM-DD
    4. 结束日期(end_date)，格式为YYYY-MM-DD
    5. 最大参与人数(max_participants)
    6. 难度级别(level)：easy, medium, hard
    7. 竞赛赛道(track)：security, development, operation, integration
    8. 竞赛规则(rules)
    9. 奖品设置(prizes)，可选

    分析用户的输入，判断哪些信息已经提供，哪些还缺失。
    如果信息不完整，友好地引导用户提供缺失的信息。
    如果信息已经完整，则生成完整的竞赛信息并返回。
    
    你需要以JSON格式输出你的分析结果，格式如下：
    ```json
    {
        "competition": {
            // 如果信息完整，这里包含完整的竞赛信息
            // 如果信息不完整，这里为null
        },
        "message": "对用户的回复消息",
        "status": "pending或complete",
        "missing_fields": ["缺失的字段列表"]
    }
    ```
    """
    
    # 将系统提示添加到消息历史
    messages = [Message.system_message(system_prompt)] + history
    
    # 调用LLM进行分析
    _, response = await llm.chat(messages=messages, stream=False, temperature=0.2)
    
    try:
        # 解析LLM返回的JSON响应
        result = parse_markdown_json(response)
        
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="无法解析LLM响应")
        
        # 构建符合API标准的响应格式
        response_data = {
            "status": result.get("status", "pending"),
            "message": result.get("message", ""),
            "missing_fields": result.get("missing_fields", []),
            "competition": result.get("competition")
        }
        
        return {
            "code": 200,
            "error": "",
            "data": response_data
        }
    except Exception as e:
        logger.error(f"创建竞赛失败: {str(e)}, 响应: {response}")
        return {
            "code": 500,
            "error": f"创建竞赛失败: {str(e)}",
            "data": {"status": "error", "message": "处理请求时发生错误"}
        }




