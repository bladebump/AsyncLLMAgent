from pydantic import BaseModel, ValidationError
from core.llms import AsyncBaseChatCOTModel
from apis.competition_utils.schema import Competition, CTFGroup, stage_map
from utils.log import logger
import re
import json

def check_item_missing_field(item: BaseModel, parent_field: str = "") -> list[str]:
    missing_fields = []
    model_class = item.__class__
    for field in model_class.model_fields.keys():
        name = f"{parent_field}.{field}" if parent_field else field
        if getattr(item, field) is None:
            missing_fields.append(f"{name}:{model_class.model_fields[field].description}")
        elif isinstance(getattr(item, field), BaseModel):
            missing_fields.extend(check_item_missing_field(getattr(item, field), parent_field=name))
        elif isinstance(getattr(item, field), list):
            for index, item in enumerate(getattr(item, field)):
                if isinstance(item, BaseModel):
                    missing_fields.extend(check_item_missing_field(item, parent_field=f"{name}[{index}]"))
    return missing_fields

async def analyze_competition_completeness(competition: Competition, llm: AsyncBaseChatCOTModel) -> tuple[str, list[str]]:
    """
    分析竞赛配置的完整性，确定下一步需要填写的内容
    
    Args:
        competition: 当前竞赛配置
        llm: 语言模型
        
    Returns:
        tuple: (下一步指导, 缺失字段列表)
    """
    # 将竞赛对象转换为字典
    competition_dict = competition.model_dump(mode='python')
    missing_fields = check_item_missing_field(competition)

    prompt = f"""
根据下面列出的竞赛配置中缺失的字段，你需要确定下一步用户填写内容，并按照严格的格式返回回答。

缺失的字段:
{missing_fields}

当前竞赛配置:
{competition_dict}

【格式要求】
请严格按照以下两种格式之一回答:

格式一：如果竞赛已完成配置，必须只返回如下文本（不要添加其他任何内容）:
竞赛配置完成

格式二：如果竞赛配置未完成，返回一个简洁的引导语句，说明下一步用户需要填写什么内容。

【判断标准】
* 如果所有必要字段都已填写（无缺失字段）且至少有一个赛程 -> 返回"竞赛配置完成"
* 如果用户明确表示完成创建 -> 返回"竞赛配置完成"
* 如果还有缺失字段 -> 返回下一步引导提示

重要提示：如果你认为配置已完成，必须严格返回"竞赛配置完成"这五个字，不要添加任何其他字符、标点或说明。
"""
    thinking, next_step = await llm.chat(prompt, stream=False)
    
    # 检查返回的内容，如果包含"竞赛配置完成"就标准化
    if "竞赛配置完成" in next_step:
        next_step = "竞赛配置完成"
    
    return next_step, missing_fields

async def process_user_input(competition: Competition, user_input: str, history: list[dict], llm: AsyncBaseChatCOTModel) -> tuple[Competition, str]:
    """
    处理用户输入并更新competition对象
    
    Args:
        competition: 当前竞赛配置
        user_input: 用户输入
        history: 对话历史
        llm: 语言模型
        
    Returns:
        tuple: (更新后的竞赛对象, 更新消息)
    """
    # 将竞赛对象转换为字典
    competition_dict = competition.model_dump(mode='python')
    # 分析当前竞赛配置的缺失字段
    missing_fields = check_item_missing_field(competition)
    # 使用LLM分析用户输入，判断用户意图并更新竞赛对象
    prompt = f"""
我需要分析用户输入，并将其映射到竞赛配置的相应字段。

当前竞赛配置:
{competition_dict}

缺失的字段:
{missing_fields}

用户输入:
{user_input}

对话历史:
{history}

任务：
1. 分析用户意图，判断用户试图填写/更新哪个字段
2. 将用户输入解析为合适的格式，并生成更新竞赛对象的操作指令
3. 格式化后的结果应该是一个JSON对象，包含以下字段:
   - "field_to_update": 需要更新的字段路径，使用点表示法，例如 "baseInfo.name" 或 "stageList[0].config.openType"
   - "update_value": 更新的值，应该是适合该字段的类型
   - "action": 更新操作类型，可以是 "update" (更新现有字段), "add" (添加新阶段), "remove" (删除阶段), "add_group" (添加CTF分组)，"add_corpus" (添加题库)
   - "description": 一句话描述更新内容

如果用户输入无法映射到任何字段或者无法确定用户意图，请返回:
{{"action": "none", "description": "无法确定用户意图"}}

请考虑以下特殊情况:
1. 如果用户想添加新阶段，需要先确定阶段类型（CTF、AWD、BTC、THEORY）:
   - 阶段类型说明：CTF（夺旗赛）、AWD（攻防赛）、BTC（闯关赛）、THEORY（理论赛）
   - update_value 只需要放入阶段类型，不需要包含其他字段
2. 如果用户想添加一个CTF的分组的话
   - update_value 只需要放入"CTF_GROUP"，不需要包含其他字段
3. 如果用户想添加一个题库的话
   - update_value 需要放入的是题目要求描述。包含题目的类型和难度相关，或者是技能要求等描述
4. 如果用户想删除阶段，应该识别要删除的阶段索引
5. 如果用户输入包含多个字段的信息，应该生成多个更新操作，格式为一个数组
6. 如果这是创建竞赛的初始阶段，用户可能会提供竞赛名称和简介等基本信息

请尽量精确解析用户意图，即使用户输入的信息不完整或者格式不规范。
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
        
        for update in updates:
            action = update.get("action", "none")
            description = update.get("description", "无法确定用户意图")
            
            if action == "none":
                update_messages.append(description)
                continue
                
            field_path = update.get("field_to_update", "")
            update_value = update.get("update_value")
            
            if action == "update":
                # 更新现有字段
                updated = update_field_by_path(competition_dict, field_path, update_value)
                if updated:
                    update_messages.append(f"已更新: {description}")
                else:
                    update_messages.append(f"更新失败: {description}")
            
            elif action == "add":
                if field_path == "stageList":
                    stage_type = update_value
                    if stage_type in stage_map:
                        stage_class = stage_map[stage_type]
                        stage_obj = stage_class()
                        competition_dict["stageList"].append(stage_obj)
                        update_messages.append(f"已添加: {description}")
                    else:
                        update_messages.append(f"添加失败: 不允许添加{stage_type}阶段")
            elif action == "add_group":
                if field_path == "groupList":
                    group_obj = CTFGroup()
                    competition_dict["groupList"].append(group_obj)
                    update_messages.append(f"已添加: {description}")
            elif action == "remove":
                if field_path.startswith("stageList[") and field_path.endswith("]"):
                    try:
                        index = int(field_path[len("stageList["):-1])
                        if 0 <= index < len(competition_dict.get("stageList", [])):
                            del competition_dict["stageList"][index]
                            update_messages.append(f"已删除: {description}")
                        else:
                            update_messages.append(f"删除失败: 索引 {index} 超出范围")
                    except ValueError:
                        update_messages.append(f"删除失败: 无效的索引")
                else:
                    update_messages.append(f"删除失败: 不允许删除字段，只能删除阶段")
            elif action == "add_corpus":
                updated = choose_corpus(competition_dict, field_path, update_value)
                if updated:
                    update_messages.append(f"已添加: {description}")
                else:
                    update_messages.append(f"添加失败: {description}")


        updated_competition = Competition.model_validate(competition_dict)
        update_message = "\n".join(update_messages)
        return updated_competition, update_message
    
    except ValidationError as e:
        logger.error(f"验证更新后的竞赛对象失败: {e}")
        return competition, f"更新失败: {e}"
            
    except Exception as e:
        logger.error(f"处理用户输入失败: {e}")
        return competition, f"处理用户输入失败: {e}"

def get_field_by_path(data: dict, path: str):
    """
    通过路径获取字段值
    
    Args:
        data: 数据字典
        path: 字段路径，例如 "baseInfo.name" 或 "stageList[0].config.openType"
        
    Returns:
        字段值或None
    """
    if not path:
        return data
    
    parts = path.split(".")
    current = data
    
    for part in parts:
        # 处理数组索引
        if "[" in part and part.endswith("]"):
            field_name, index_str = part.split("[", 1)
            index = int(index_str[:-1])
            
            if field_name not in current:
                return None
            if not isinstance(current[field_name], list) or index >= len(current[field_name]):
                return None
            
            current = current[field_name][index]
        else:
            if part not in current:
                return None
            current = current[part]
    
    return current

def update_field_by_path(data: dict, path: str, value) -> bool:
    """
    通过路径更新字段值
    
    Args:
        data: 数据字典
        path: 字段路径，例如 "baseInfo.name" 或 "stageList[0].config.openType"
        value: 新值
        
    Returns:
        是否更新成功
    """
    parts = path.split(".")
    current = data
    
    # 处理所有中间路径
    for i, part in enumerate(parts[:-1]):
        # 处理数组索引
        if "[" in part and part.endswith("]"):
            field_name, index_str = part.split("[", 1)
            index = int(index_str[:-1])
            
            if field_name not in current:
                current[field_name] = []
            if not isinstance(current[field_name], list):
                return False
            
            # 扩展列表长度如果需要
            while len(current[field_name]) <= index:
                current[field_name].append({})
            
            current = current[field_name][index]
        else:
            if part not in current:
                current[part] = {}
            if not isinstance(current[part], dict):
                return False
            current = current[part]
    
    # 处理最后一个部分
    last_part = parts[-1]
    
    # 处理数组索引
    if "[" in last_part and last_part.endswith("]"):
        field_name, index_str = last_part.split("[", 1)
        index = int(index_str[:-1])
        
        if field_name not in current:
            current[field_name] = []
        if not isinstance(current[field_name], list):
            return False
        
        # 扩展列表长度如果需要
        while len(current[field_name]) <= index:
            current[field_name].append(None)
        
        current[field_name][index] = value
    else:
        current[last_part] = value
    
    return True

def choose_corpus(competition_dict: dict, field_path: str, update_value: str) -> bool:
    """
    选择题库
    """
    logger.debug(f"选择题库: {update_value}")
    return True
    

