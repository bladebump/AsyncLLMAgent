from fastapi import APIRouter, Depends
from pydantic import BaseModel
from core.llms import AsyncBaseChatCOTModel
from .utils import get_llm, get_llm_cot
from utils.log import logger

competition_router = APIRouter(prefix="/competition")

class WPFormat(BaseModel):
    wp: str
    format_yaml: str
    use_cot_model: bool = False

@competition_router.post("/wp_format")
async def wp_format(wp_format: WPFormat, llm: AsyncBaseChatCOTModel = Depends(get_llm), cot_llm: AsyncBaseChatCOTModel = Depends(get_llm_cot)):
    """
    格式化WP
    """
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
    thinking, response = await llm.chat(prompt,stream=False)
    return {"code": 200, "error": None, "data": {"wp": response, "thinking": thinking}}
