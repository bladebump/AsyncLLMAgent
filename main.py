from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.llms.openai_llm import OpenAICoT
from core.embeddings.silicon_agent import SiliconEmbeddingAgent
from core.ranks import SiliconRankAgent
from core.vector.milvus import MilvusVectorStore
from core.config import config
from apis import all_routers
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_provider = config.current_provider
    llm_config = config.llm_providers[llm_provider]
    app.state.llm = OpenAICoT(
        api_base=llm_config.api_base,
        api_key=llm_config.api_key,
        model=llm_config.model,
        support_fn_call=True
    )
    app.state.llm_cot = OpenAICoT(
        api_base=config.llm_cot.api_base,
        api_key=config.llm_cot.api_key,
        model=config.llm_cot.model,
        support_fn_call=False
    )
    app.state.embedding = SiliconEmbeddingAgent(
        url=config.embedding.api_base,
        api_key=config.embedding.api_key,
        model=config.embedding.model,
    )
    app.state.reranker = SiliconRankAgent(
        url=config.reranker.api_base,
        api_key=config.reranker.api_key,
        model=config.reranker.model,
    )
    if config.milvus.enable:
        milvus = MilvusVectorStore(
        uri=config.milvus.uri,
        username=config.milvus.username,
        password=config.milvus.password,
        dense_vector_dim=config.milvus.dense_vector_dim,
            use_sparse_vector=config.milvus.use_sparse_vector
        )
    else:
        milvus = None
    app.state.milvus_store = milvus
    yield
    if config.milvus.enable:
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
