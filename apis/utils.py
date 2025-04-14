from fastapi import Request
import json5
import json
import re
import yaml

def get_llm(request: Request):
    return request.app.state.llm

def get_llm_cot(request: Request):
    return request.app.state.llm_cot

def get_embedding(request: Request):
    return request.app.state.embedding

def get_milvus_store(request: Request):
    return request.app.state.milvus_store

def get_reranker(request: Request):
    return request.app.state.reranker

def parse_markdown_json(text: str) -> any:
    """解析可能被markdown代码块包裹的JSON字符串
    
    Args:
        text: 输入的文本，可能包含markdown代码块
        
    Returns:
        解析后的JSON对象
        
    Raises:
        json.JSONDecodeError: 当JSON解析失败时
    """
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    return json5.loads(text.strip())

def parse_markdown_yaml(text: str) -> any:
    """从Markdown文本中提取YAML内容并解析
    
    Args:
        text: 输入的文本，可能包含markdown代码块
        
    Returns:
        解析后的YAML对象，如果为None则返回空列表
        
    Raises:
        yaml.YAMLError: 当YAML解析失败时
    """
    # 尝试匹配```yaml和```之间的内容
    yaml_pattern = r"```yaml\s*([\s\S]*?)\s*```"
    yaml_match = re.search(yaml_pattern, text)
    
    if yaml_match:
        yaml_content = yaml_match.group(1)
    else:
        # 如果没有特定的yaml标记，尝试匹配普通的```和```之间的内容
        code_pattern = r"```\s*([\s\S]*?)\s*```"
        code_match = re.search(code_pattern, text)
        if code_match:
            yaml_content = code_match.group(1)
        else:
            # 如果没有任何代码块标记，使用整个文本
            yaml_content = text
    
    # 解析YAML内容
    result = yaml.safe_load(yaml_content)
    return [] if result is None else result