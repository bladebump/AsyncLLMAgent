from fastapi import Request

def get_llm(request: Request):
    return request.app.state.llm

def get_llm_cot(request: Request):
    return request.app.state.llm_cot

def get_embedding(request: Request):
    return request.app.state.embedding

def get_milvus_store(request: Request):
    return request.app.state.milvus_store
