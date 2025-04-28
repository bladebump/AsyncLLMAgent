from pydantic import BaseModel, ValidationError
from core.llms import AsyncBaseChatCOTModel
from apis.course_utils.schema import Course, CourseBaseInfo, DifficultyType
from utils.log import logger
from apis.utils import update_field_by_path, get_field_by_path
import re
import json

def check_item_missing_field(item: BaseModel, parent_field: str = "") -> list[str]:
    """
    检查对象中缺失的字段

    Args:
        item: 要检查的对象
        parent_field: 父字段名，用于构建完整路径
    
    Returns:
        list: 缺失字段的列表，每个元素格式为 "字段路径:字段描述"
    """
    missing_fields = []
    model_class = item.__class__
    for field in model_class.model_fields.keys():
        name = f"{parent_field}.{field}" if parent_field else field
        field_value = getattr(item, field)
        
        if field_value is None:
            missing_fields.append(f"{name}:{model_class.model_fields[field].description}")
        elif isinstance(field_value, BaseModel):
            missing_fields.extend(check_item_missing_field(field_value, parent_field=name))
        elif isinstance(field_value, list):
            for index, list_item in enumerate(field_value):
                if isinstance(list_item, BaseModel):
                    missing_fields.extend(check_item_missing_field(list_item, parent_field=f"{name}[{index}]"))
                # 对于非 BaseModel 类型的列表元素，不进行检查
                
    return missing_fields

async def analyze_course_completeness(course: Course, user_input: str, llm: AsyncBaseChatCOTModel) -> tuple[str, list[str]]:
    """
    分析课程配置的完整性，确定下一步需要填写的内容

    Args:
        course: 当前课程配置
        llm: 语言模型
        
    Returns:
        tuple: (下一步指导, 缺失字段列表)
    """
    course_dict = course.model_dump(mode='python')
    missing_fields = check_item_missing_field(course)
    prompt = f"""
根据下面列出的课程配置中缺失的字段，你需要确定下一步用户填写内容，并按照严格的格式返回回答。

用户输入:
{user_input}

缺失的字段:
{missing_fields}

当前课程    配置:
{course_dict}

【回答要求】
请严格按照以下规则回答:

1. 只有在用户明确表示完成创建（如"创建完成"、"确认无误"等）时，才返回"课程配置完成"这五个字，不添加任何其他内容
2. 在其他情况下（即使没有缺失字段），返回简洁的下一步引导提示，可以是:
   - 需要填写的缺失字段信息
   - 可添加/修改的内容建议
   - 询问用户是否确认完成创建

注意: 不要仅因为没有缺失字段就判定为完成，必须用户明确确认才返回"课程配置完成"。
"""
    thinking, next_step = await llm.chat(prompt, stream=False)
    
    # 检查返回的内容，如果包含"竞赛配置完成"就标准化
    if "课程配置完成" in next_step:
        next_step = "课程配置完成"
    
    return next_step, missing_fields

async def process_user_input(course: Course, user_input: str, history: list[dict], llm: AsyncBaseChatCOTModel) -> tuple[Course, str]:
    """
    处理用户输入并更新course对象

    Args:
        course: 当前课程配置
        user_input: 用户输入
        history: 对话历史
        llm: 语言模型
    
    Returns:
        tuple: (更新后的课程对象, 更新消息)
    """
    
    course_dict = course.model_dump(mode='python')
    missing_fields = check_item_missing_field(course)
    prompt = f"""
我需要分析用户输入，并将其映射到课程配置的相应字段。

当前课程配置:
{course_dict}

缺失的字段:
{missing_fields}

用户输入:
{user_input}

对话历史:
{history}

【任务】
分析用户意图，将用户输入解析为适当的课程配置更新操作。

【输出格式】
返回一个JSON格式的操作指令，包含以下字段:
- "field_to_update": 需要更新的字段路径，使用点表示法（如"baseInfo.name"）
- "update_value": 更新的值
- "action": 操作类型，只能是 "update"
- "description": 简短描述此次更新内容

【特别说明】
1. difficulty 字段必须是字符串类型，且只能是 '1'、'2' 或 '3'
2. tags 和 relatedCourses 字段必须是整数列表
3. authors 和 knowledgePoints 字段必须是字符串列表
4. 所有列表类型的字段必须确保是列表格式，不能是单个值

如果用户意图不明确，请返回:
{{"action": "none", "description": "无法确定用户意图"}}

请尽量准确解析用户意图，即使用户输入格式不规范或信息不完整。
"""
    
    thinking, parse_result = await llm.chat(prompt, stream=False)
    logger.debug(f"解析用户输入结果: {parse_result}")
    
    try:
        # 使用正则表达式提取JSON部分
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```|```([\s\S]*?)```|({[\s\S]*})', parse_result)
        if json_match:
            json_str = json_match.group(1) or json_match.group(2) or json_match.group(3)
            updates = json.loads(json_str)
        else:
            updates = json.loads(parse_result)
        
        # 确保updates是列表
        if not isinstance(updates, list):
            updates = [updates]
        
        update_messages = []
        course_dict = course.model_dump(mode='python')
        logger.info(f"更新前的课程配置: {course_dict}")
        
        for update in updates:
            action = update.get("action", "none")
            description = update.get("description", "无法确定用户意图")
            
            if action == "none":
                update_messages.append(description)
                continue
                
            field_path = update.get("field_to_update", "")
            update_value = update.get("update_value")
            
            if not field_path or update_value is None:
                update_messages.append(f"更新失败: 缺少必要的字段路径或更新值")
                continue
                
            # 特殊处理 difficulty 字段
            if field_path == "baseInfo.difficulty":
                if isinstance(update_value, (int, float)):
                    update_value = str(int(update_value))
                if update_value not in ['1', '2', '3']:
                    update_messages.append(f"更新失败: difficulty 必须是 '1'、'2' 或 '3'")
                    continue
                    
            # 特殊处理列表类型字段
            if field_path in ["baseInfo.authors", "baseInfo.knowledgePoints"]:
                if not isinstance(update_value, list):
                    update_value = [str(update_value)]
                else:
                    update_value = [str(item) for item in update_value]
                    
            if field_path in ["baseInfo.tags", "baseInfo.relatedCourses"]:
                if not isinstance(update_value, list):
                    update_value = [int(update_value)]
                else:
                    try:
                        update_value = [int(item) for item in update_value]
                    except (ValueError, TypeError):
                        update_messages.append(f"更新失败: {field_path} 必须是整数列表")
                        continue
                
            if action == "update":
                try:
                    logger.info(f"尝试更新字段: {field_path} = {update_value}")
                    updated = update_field_by_path(course_dict, field_path, update_value)
                    logger.info(f"更新结果: {updated}")
                    logger.info(f"更新后的课程配置: {course_dict}")
                    
                    if updated:
                        update_messages.append(f"已更新: {description}")
                    else:
                        update_messages.append(f"更新失败: 字段路径 {field_path} 不存在")
                except Exception as e:
                    logger.error(f"更新字段时发生错误: {str(e)}")
                    update_messages.append(f"更新失败: {str(e)}")
            else:
                update_messages.append(f"不支持的操作类型: {action}")

        try:
            logger.info(f"验证更新后的课程配置: {course_dict}")
            updated_course = Course.model_validate(course_dict)
            logger.info(f"验证成功，新的课程对象: {updated_course}")
            update_message = "\n".join(update_messages)
            return updated_course, update_message
        except ValidationError as e:
            logger.error(f"验证更新后的课程对象失败: {e}")
            return course, f"更新失败: 课程数据验证错误 - {str(e)}"
    
    except json.JSONDecodeError as e:
        logger.error(f"解析JSON失败: {e}")
        return course, f"解析用户输入失败: JSON格式错误"
            
    except Exception as e:
        logger.error(f"处理用户输入失败: {e}")
        return course, f"处理用户输入失败: {str(e)}"