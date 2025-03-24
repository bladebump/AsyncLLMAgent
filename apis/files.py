from fastapi import APIRouter, UploadFile, File, Depends, Form
from .utils import *
from core.vector.base import VectorStoreBase
from core.embeddings.base import EmbeddingAgent
from pydantic import BaseModel
from core.vector.base import Document
from core.spliter.text import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

files_router = APIRouter(prefix="/files")

class CreateCollection(BaseModel):
    collection_name: str

@files_router.post("create_collection")
async def create_collection(collection: CreateCollection, milvus: VectorStoreBase = Depends(get_milvus_store)):
    return await milvus.create_collection(collection.collection_name)

@files_router.post("upload_file")
async def upload_file(file: UploadFile = File(...), 
                      collection_name: str = Form(...),
                      milvus: VectorStoreBase = Depends(get_milvus_store), 
                      embedding: EmbeddingAgent = Depends(get_embedding)):
    content = await file.read()
    filename = file.filename
    department = file.content_type
    text = await embedding.embed_text(content)
    doc_list = []
    if len(text) > CHUNK_SIZE:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        text_list = splitter.split_text(text)
        for text in text_list:
            doc_list.append(Document(text=text, filename=filename, department=department))
    else:
        doc_list = [Document(text=text, filename=filename, department=department)]
    return await milvus.add_batch(doc_list, collection_name)