from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.llms.openai_llm import OpenAICoT
from core.embeddings.silicon_agent import SiliconEmbeddingAgent
from core.vector.milvus import MilvusVectorStore
from config import *
from apis import all_routers
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = OpenAICoT(
        api_base=LLM_API_BASE,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        support_fn_call=True
    )
    app.state.llm_cot = OpenAICoT(
        api_base=LLM_COT_API_BASE,
        api_key=LLM_COT_API_KEY,
        model=LLM_COT_MODEL,
        support_fn_call=False
    )
    app.state.embedding = SiliconEmbeddingAgent(
        url=EMBEDDING_API_BASE,
        api_key=EMBEDDING_API_KEY,
        model=EMBEDDING_MODEL,
    )
    app.state.milvus_store = MilvusVectorStore(
        uri=MILVUS_URI,
        username=MILVUS_USERNAME,
        password=MILVUS_PASSWORD,
        dense_vector_dim=MILVUS_DENSE_VECTOR_DIM,
        use_sparse_vector=MILVUS_USE_SPARSE_VECTOR
    )
    yield
    await app.state.milvus_store.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in all_routers:
    app.include_router(router)

@app.get("/")
async def root():
    return {"message": "llm is ready"}

@app.get("/health")
async def health():
    return {"message": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5678)
