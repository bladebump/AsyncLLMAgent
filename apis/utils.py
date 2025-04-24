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