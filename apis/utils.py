from fastapi import Request
import json
def get_llm(request: Request):
    return request.app.state.llm

def get_llm_cot(request: Request):
    return request.app.state.llm_cot

def get_embedding(request: Request):
    return request.app.state.embedding

def get_milvus_store(request: Request):
    return request.app.state.milvus_store

def parse_markdown_json(text: str) -> dict:
    """解析可能被markdown代码块包裹的JSON字符串
    
    Args:
        text: 输入的文本，可能包含markdown代码块
        
    Returns:
        解析后的JSON对象
        
    Raises:
        json.JSONDecodeError: 当JSON解析失败时
    """
    # 移除markdown代码块
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())