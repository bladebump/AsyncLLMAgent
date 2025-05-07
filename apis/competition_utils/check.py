from pydantic import BaseModel, ValidationError
from core.llms import AsyncBaseChatCOTModel
from apis.competition_utils.schema import Competition, CTFGroup, stage_map
from apis.utils import update_field_by_path, get_field_by_path, check_item_missing_field, parse_markdown_json
from utils.log import logger
from core.config import config
import re
import json
import random
import httpx

async def analyze_competition_completeness(competition: Competition, user_input: str, llm: AsyncBaseChatCOTModel) -> tuple[str, list[str]]:
    """分析竞赛配置的完整性，确定下一步需要填写的内容"""
    # 将竞赛对象转换为字典
    competition_dict = competition.model_dump(mode='python')
    missing_fields = check_item_missing_field(competition)

    prompt = f"""
根据下面列出的竞赛配置中缺失的字段，你需要确定下一步用户填写内容，并按照严格的格式返回回答。

<用户输入>
{user_input}
</用户输入>

<缺失的字段>
{missing_fields}
</缺失的字段>

<当前竞赛配置>
{competition_dict}
</当前竞赛配置>

【回答要求】
请严格按照以下规则回答:

1. 只有在用户明确表示完成创建（如"创建完成"、"确认无误"等）时，才返回"竞赛配置完成"这五个字，不添加任何其他内容
2. 在其他情况下（即使没有缺失字段），返回简洁的下一步引导提示，可以是:
   - 需要填写的缺失字段信息
   - 可添加/修改的内容建议
   - 询问用户是否确认完成创建
3. 请按照缺失字段的顺序引导用户填写，不要跳过任何字段。

注意: 不要仅因为没有缺失字段就判定为完成，必须用户明确确认才返回"竞赛配置完成"。
"""
    thinking, next_step, _ = await llm.chat(prompt, stream=False)
    
    # 检查返回的内容，如果包含"竞赛配置完成"就标准化
    if "竞赛配置完成" in next_step:
        next_step = "竞赛配置完成"
    
    return next_step, missing_fields

async def process_user_input(competition: Competition, user_input: str, history: list[dict], llm: AsyncBaseChatCOTModel, token: str) -> tuple[Competition, str]:
    """处理用户输入并更新competition对象"""
    # 将竞赛对象转换为字典
    competition_dict = competition.model_dump(mode='python')
    # 分析当前竞赛配置的缺失字段
    missing_fields = check_item_missing_field(competition)
    # 使用LLM分析用户输入，判断用户意图并更新竞赛对象
    prompt = f"""
我需要分析用户输入，并将其映射到竞赛配置的相应字段。

<当前竞赛配置>
{competition_dict}
</当前竞赛配置>

<缺失的字段>
{missing_fields}
</缺失的字段>

<用户输入>
{user_input}
</用户输入>

【任务】
分析用户意图，将用户输入解析为适当的竞赛配置更新操作。

【输出格式】
返回一个JSON格式的操作指令，包含以下字段:
- "field_to_update": 需要更新的字段路径，使用点表示法（如"baseInfo.name"或"stageList[0].config.openType"）
- "update_value": 更新的值
- "action": 操作类型，可选值:
  * "update": 更新现有字段
  * "add": 添加新阶段
  * "remove": 删除阶段
  * "add_group": 添加CTF分组
  * "corpus_choice": 添加或者修改赛题设置
- "description": 简短描述此次更新内容

如果用户意图不明确，请返回:
{{"action": "none", "description": "无法确定用户意图"}}

【特殊情况处理】
1. 添加新阶段(action="add")
   - 阶段类型: CTF(夺旗赛)、AWD(攻防赛)、BTC(闯关赛)、THEORY(理论赛)
   - update_value 为一个列表,比如["CTF", "AWD", "BTC"]

2. 添加CTF分组(action="add_group")
   - 需要识别用户输入的CTF分组名称，比如"WEB"、"RE"、"PWN"
   - update_value为一个列表，比如["WEB", "RE", "PWN"]

3. 修改题目(action="corpus_choice")
   - update_value为一个列表，列表中每个元素的格式如下:
     [{{"mode": "CTF|AWD|BTC", "difficulty": "VERY_EASY|EASY|MEDIUM|HARD|VERY_HARD", "classify": "WEB|MISC|CRYPTO|REVERSE|PWN", "num": 题目数量}}]

4. 删除阶段(action="remove")
   - 需识别要删除的阶段索引

5. 多字段更新
   - 如用户输入涉及多个字段，请返回包含多个更新操作的数组

请尽量准确解析用户意图，即使用户输入格式不规范或信息不完整。
"""
    
    thinking, parse_result, _ = await llm.chat(prompt, stream=False)
    logger.debug(f"解析用户输入结果: {parse_result}")
    
    try:
        updates = parse_markdown_json(parse_result)
        
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
            update_value = update.get("update_value", None)
            
            if action == "update":
                # 更新现有字段
                updated = update_field_by_path(competition_dict, field_path, update_value)
                if updated:
                    update_messages.append(f"已更新: {description}")
                else:
                    update_messages.append(f"更新失败: {description}")
            
            elif action == "add":
                if field_path == "stageList":
                    for stage_type in update_value:
                        if stage_type in stage_map:
                            stage_class = stage_map[stage_type]
                            stage_obj = stage_class()
                            competition_dict["stageList"].append(stage_obj)
                            update_messages.append(f"已添加: {description}")
                        else:
                            update_messages.append(f"添加失败: 不允许添加{stage_type}阶段")
            elif action == "add_group":
                if field_path.endswith(".groupList"):
                    if not isinstance(update_value, list):
                        update_value = [update_value]
                    for item in update_value:
                        parent_path = field_path[:field_path.rfind(".")]
                        group_obj = CTFGroup(name=item)
                        current = get_field_by_path(competition_dict, parent_path)
                        if current is None:
                            update_messages.append(f"添加失败: 未能找到{parent_path}字段")
                        else:
                            if current["groupList"] is None:
                                current["groupList"] = []
                            current["groupList"].append(group_obj)
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
            elif action == "corpus_choice":
                updated, error_message = await choose_corpus(competition_dict, field_path, update_value, llm, token)
                if updated:
                    update_messages.append(f"已添加: {description} \n{error_message}")
                else:
                    update_messages.append(f"添加失败: {error_message}")


        updated_competition = Competition.model_validate(competition_dict)
        update_message = "\n".join(update_messages)
        return updated_competition, update_message
    
    except ValidationError as e:
        logger.error(f"验证更新后的竞赛对象失败: {e}")
        return competition, f"更新失败: {e}"
            
    except Exception as e:
        logger.error(f"处理用户输入失败: {e}")
        return competition, f"处理用户输入失败: {e}"

async def get_corpus_data(token: str, corpus_type: str, difficulty: str|None, classify: str|None) -> dict:
    """获取平台的题目信息"""
    if not (corpus_type in ["CTF", "AWD", "BTC"] or corpus_type in ["ctf", "awd", "btc"]):
        raise ValueError("corpus_type 必须是 CTF, AWD, BTC 或 ctf, awd, btc")
    url = f"{config.platform.url}/slab-match/api/v1/corpus/{corpus_type.lower()}/list"
    headers = {
        "Authorization": token
    }
    params = {
        "page": 1,
        "size": 100,
    }
    if difficulty:
        params["difficulty"] = difficulty
    if classify:
        params["classify"] = classify
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        response_data = response.json()
        corpus_data = []
        if "tbody" not in response_data["data"]:
            raise ValueError(f"获取题库失败: {response_data['message']}")
        for item in response_data["data"]["tbody"]:
            corpus_data.append(item['id'])
        return corpus_data

async def choose_corpus(competition_dict: dict, field_path: str, update_value: str, llm: AsyncBaseChatCOTModel, token: str) -> tuple[bool, str]:
    """选择题库"""
    parent_path = field_path[:field_path.rfind(".")]
    current = get_field_by_path(competition_dict, parent_path)
    corpus_list = []
    error_message = ""
    try:
        if isinstance(update_value, list):
            for item in update_value:
                difficulty = item.get("difficulty")
                classify = item.get("classify")
                mode = item.get("mode")
                num = item.get("num")
                corpus_data = await get_corpus_data(token, mode, difficulty, classify)
                if len(corpus_data) == 0:
                    error_message += f"难度{difficulty}分类{classify}题库为空, 取消难度随机选择\n"
                    corpus_data = await get_corpus_data(token, mode, None, classify)
                if len(corpus_data) == 0:
                    error_message += f"分类{classify}题库为空, 取消分类随机选择\n"
                    corpus_data = await get_corpus_data(token, mode, None, None)
                corpus_list.extend(random.sample(corpus_data, num))
        current["corpusId"] = corpus_list
        return True, error_message
    except Exception as e:
        logger.error(f"选择题库失败: {e}")
        error_message += f"选择题库失败: {e}"
        return False, error_message
    
