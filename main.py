from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.llms import OpenAICoT, QwenCoT
from core.embeddings.silicon_agent import SiliconEmbeddingAgent
from core.ranks import SiliconRankAgent
from core.vector.milvus import MilvusVectorStore
from core.config import config
from apis import all_routers
from fastapi.middleware.cors import CORSMiddleware

llm_map = {
    "openai": OpenAICoT,
    "qwen": QwenCoT
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm_list = {}
    app.state.llm_cot = {}
    for provider_name in config.llm_providers:
        provider_config = config.llm_providers[provider_name]
        app.state.llm_list[provider_name] = llm_map[provider_config.model_base](
            api_base=provider_config.api_base,
            api_key=provider_config.api_key,
            model=provider_config.model,
            support_fn_call=True,
            enable_thinking=False
        )
    for provider_name in config.llm_cot_providers:
        provider_config = config.llm_cot_providers[provider_name]
        app.state.llm_cot[provider_name] = llm_map[provider_config.model_base](
            api_base=provider_config.api_base,
            api_key=provider_config.api_key,
            model=provider_config.model,
            support_fn_call=True,
            enable_thinking=True
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
    uvicorn.run(app, host="127.0.0.1", port=5678)
