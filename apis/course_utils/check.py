from pydantic import ValidationError
from core.llms import AsyncBaseChatCOTModel
from .schema import Course
from utils.log import logger
from apis.utils import update_field_by_path, check_item_missing_field, parse_markdown_yaml, parse_markdown_json
from core.config import config
import httpx
from copy import deepcopy
from core.schema import Message

async def analyze_course_completeness(course: Course, user_input: str, llm: AsyncBaseChatCOTModel) -> tuple[str, list[str]]:
    """分析课程配置的完整性，确定下一步需要填写的内容"""
    course_dict = course.model_dump(mode='python')
    missing_fields = check_item_missing_field(course)
    prompt = f"""
根据下面列出的课程配置中缺失的字段，你需要确定下一步用户填写内容，并按照格式返回回答。

<用户输入>
{user_input}
</用户输入>

<缺失的字段>
{missing_fields}
</缺失的字段>

<当前课程配置>
{course_dict}
</当前课程配置>

【回答要求】
请严格按照以下规则回答:

1. 在用户明确表示完成创建时，返回"课程配置完成"这五个字，不添加任何其他内容
2. 在其他情况下，返回简洁的下一步引导提示，可以是:
   - 需要填写的缺失字段信息
   - 可添加/修改的内容建议
   - 询问用户是否确认完成创建
"""
    thinking, next_step, _ = await llm.chat(prompt, stream=False, temperature=0.01)
    
    # 检查返回的内容，如果包含"竞赛配置完成"就标准化
    if "课程配置完成" in next_step:
        next_step = "课程配置完成"
    
    return next_step, missing_fields

async def process_user_input(course: Course, user_input: str, history: list[dict], llm: AsyncBaseChatCOTModel, token: str) -> tuple[Course, str]:
    """处理用户输入并更新course对象"""
    
    course_dict = course.model_dump(mode='python')
    missing_fields = check_item_missing_field(course)
    prompt = f"""
我需要分析用户输入，并将其映射到课程配置的相应字段。

<当前课程配置>
{course_dict}
</当前课程配置>

<缺失的字段>
{missing_fields}
</缺失的字段>

<用户输入>
{user_input}
</用户输入>

【任务】
分析用户意图，将用户输入解析为适当的课程配置更新操作。

【输出格式】
返回一个JSON格式的操作指令，包含以下字段:
- "field_to_update": 需要更新的字段路径，使用点表示法（如"baseInfo.name"或"stageList[0].config.openType"）
- "update_value": 更新的值
- "action": 操作类型，可选值:
  * "update": 更新现有字段
  * "add_chapter": 添加新章节
- "description": 简短描述此次更新内容

如果用户意图不明确，请返回:
[{{"action": "none", "description": "无法确定用户意图"}}]

【返回样列】
[{{"action": "update", "field_to_update": "baseInfo.name", "update_value": "课程名称", "description": "更新课程名称"}}]

【特殊情况处理】
1. 添加新章节(action="add_chapter")
   - update_value 为用户的需求
   - 只允许添加章节，不允许修改章节
   - 只能调用add_chapter函数添加章节
   - 不允许update chapterList

请尽量准确解析用户意图，即使用户输入格式不规范或信息不完整。
"""
    
    messages = deepcopy(history)
    messages.append(Message.user_message(prompt))
    thinking, parse_result, _ = await llm.chat(messages=messages, stream=False, temperature=0.01)
    logger.debug(f"解析用户输入结果: {parse_result}")
    
    try:
        updates = parse_markdown_json(parse_result)
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
            update_value = update.get("update_value", None)

            if action == "update":
                # 更新现有字段
                updated = update_field_by_path(course_dict, field_path, update_value)
                if updated:
                    update_messages.append(f"已更新: {description}")
                else:
                    update_messages.append(f"更新失败: {description}")
            elif action == "add_chapter":
                course_dict, success, error_message = await add_chapter(course_dict, description, llm, token)
                if success:
                    update_messages.append(f"已添加章节: {description}")
                else:
                    update_messages.append(f"添加章节失败: {error_message}")

        updated_course = Course.model_validate(course_dict)
        update_message = "\n".join(update_messages)
        return updated_course, update_message
    
    except ValidationError as e:
        logger.error(f"验证更新后的课程对象失败: {e}")
        return course, f"更新失败: {e}"
    except Exception as e:
        logger.error(f"处理用户输入失败: {e}")
        return course, f"处理用户输入失败: {str(e)}"

async def get_corpus_data(token: str, keyword: str) -> dict:
    """获取课程数据"""
    url = f"{config.platform.url}/range-resource/api/knowledge/user/lists"
    params = {
        "page": 1,
        "size": 100,
        "inPermit": True,
        "search": keyword
    }
    headers = {
        "Authorization": token
    }
    result_data = []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)
            response_data = response.json()
        courses_list = response_data["data"]["tbody"]
        for course in courses_list:
            result_data.append({
                "id": course["id"],
                "level": course["level"],
                "name": course["name"],
                "summary": course["summary"],
                "classHour": course["classHour"]
            })
        return result_data, ""
    except Exception as e:
        logger.error(f"获取知识点失败: {e}")
        return [], f"获取知识点失败: {e}"

async def generate_keyword_to_get_corpus(course_dict: dict, llm: AsyncBaseChatCOTModel, token: str) -> str:
    """生成课程关键词"""
    prompt = f"""当前有这么一个课程信息
{course_dict}

请根据课程信息，生成三个关键词，用于获取知识点。
只需要返回关键词，不要添加任何其他内容。关键词之间用英文逗号隔开。

【样列】
web安全,sql注入,xss攻击

请生成：
"""
    thinking, result, _ = await llm.chat(prompt, stream=False, temperature=0.01)
    keywords = result.split(",")
    course_list = []
    error_message = ""
    for keyword in keywords:
        course_list_, error_message = await get_corpus_data(token, keyword)
        if error_message:
            error_message += f"获取知识点失败: {error_message}\n"
        else:
            course_list.extend(course_list_)
    return course_list, error_message

async def add_chapter(course_dict: dict, update_value: str, llm: AsyncBaseChatCOTModel, token: str) -> tuple[dict, bool, str]:
    """添加章节到课程中"""
    course_list, error_message = await generate_keyword_to_get_corpus(course_dict, llm, token)
    if error_message:
        return course_dict, False, error_message
    
    prompt = f"""
根据用户需求和可用知识点，生成一个课程章节列表。

【课程信息】
{course_dict}

【用户需求】
{update_value}

【可用知识点】
{course_list}

【任务】
生成一个包含多个章节的列表，每个章节需包含:
- 章节名称(name)
- 章节简介(profile)
- 相关知识点ID列表(knowledgePointList)

【严格课时控制】
1. 从用户需求中提取出总课时数要求
2. 仔细计算所选知识点的课时总和，每个知识点的课时为其classHour值
3. 确保所有选择的知识点课时总和不超过用户要求的总课时数
4. 如果可用知识点的课时总和超出限制，请进行以下优化:
   - 优先选择核心知识点
   - 减少每个章节的知识点数量
   - 确保章节之间的课时分配合理平衡

【注意点】
1. 章节名称需要与用户需求和可用知识点相匹配
2. 每个知识点的课时由其classHour字段决定
3. 总课时必须严格控制在用户要求范围内
4. 章节数目不宜过多，一般为3-5个章节
5. 在生成前，请计算选择的知识点的总课时，确保不超标
6. 请确保选择的知识点id是正确的
7. 如果课程数量不足，请选择少的课程，不要瞎编

【输出格式】
请使用YAML格式返回章节列表，并在开头注明总课时计算:

```yaml
# 总课时计算: X课时 (不超过用户要求的Y课时)
- name: 章节名称1
  profile: 章节简介内容
  knowledgePointList: 
    - 知识点ID1  # 课时:Z
    - 知识点ID2  # 课时:Z
- name: 章节名称2
  profile: 章节简介内容
  knowledgePointList:
    - 知识点ID3  # 课时:Z
    - 知识点ID4  # 课时:Z
```

请确保总课时不超过用户要求，同时保持章节内容与用户需求相匹配。
"""
    
    thinking, yaml_result, _ = await llm.chat(prompt, stream=False, temperature=0.01)
    chapters = parse_markdown_yaml(yaml_result)
    if not chapters:
        return course_dict, False, "生成的章节格式不正确，应为列表"
    course_dict["chapterList"] = chapters
    return course_dict, True, ""


async def get_tags(token: str) -> list[dict]:
    """获取平台课程标签"""
    url = f"{config.platform.url}/range-tag/api/selection/tag-select-list?service=range-edu&module=course"
    headers = {
        "Authorization": token
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response_data = response.json()
            tags_data = []
            for tag in response_data["data"]:
                if not tag['disabled']:
                    tags_data.append({
                        "value": tag["value"],
                        "label": tag["label"]
                    })
            return tags_data
    except Exception as e:
        logger.error(f"获取标签失败: {e}")
        return []
