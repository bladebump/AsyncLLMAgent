from fastapi import APIRouter, UploadFile, File, Depends, Form
from .utils import *
from core.vector.base import VectorStoreBase
from core.embeddings.base import EmbeddingAgent
from pydantic import BaseModel
from core.vector.base import Document
from core.spliter.text import RecursiveCharacterTextSplitter
from core.config import config

files_router = APIRouter(prefix="/files")

class CreateCollection(BaseModel):
    collection_name: str

@files_router.post("/create_collection")
async def create_collection(collection: CreateCollection, milvus: VectorStoreBase = Depends(get_milvus_store)):
    if milvus is None:
        return {"message": "Collection 创建失败", "error": "Milvus 未启用"}
    result = await milvus.create_collection(collection.collection_name)
    if result:
        return {"message": "Collection 创建成功", "error": ""}
    else:
        return {"message": "Collection 创建失败", "error": "Collection 已存在"}

@files_router.get("/delete_collection")
async def delete_collection(collection_name: str, milvus: VectorStoreBase = Depends(get_milvus_store)):
    if milvus is None:
        return {"message": "Collection 删除失败", "error": "Milvus 未启用"}
    await milvus.drop_collection(collection_name)
    return {"message": "Collection 删除成功", "error": ""}

@files_router.post("/upload_file")
async def upload_file(file: UploadFile = File(...), 
                      collection_name: str = Form(...),
                      department: int = Form(default=0),
                      milvus: VectorStoreBase = Depends(get_milvus_store), 
                      embedding: EmbeddingAgent = Depends(get_embedding)):
    if milvus is None:
        return {"message": "文件上传失败", "error": "Milvus 未启用"}
    content = await file.read()
    filename = file.filename
    doc_list = []
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk.size, chunk_overlap=config.chunk.overlap)
    text_list = splitter.split_text(content)
    
    for text in text_list:
        embedding_vector = await embedding.encode(text)
        doc_list.append(Document(text=text, filename=filename, department=department, dense_vector=embedding_vector))
    
    result = await milvus.add_batch(doc_list, collection_name)
    if result:
        return {"message": "文件上传成功", "error": ""}
    else:
        return {"message": "文件上传失败", "error": "文件上传失败"}
