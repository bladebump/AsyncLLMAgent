from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.llms.openai_llm import OpenAICoT
from core.embeddings.silicon_agent import SiliconEmbeddingAgent
from config import *
from apis import all_routers

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = OpenAICoT(
        api_base=LLM_API_BASE,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
    )
    app.state.llm_cot = OpenAICoT(
        api_base=LLM_COT_API_BASE,
        api_key=LLM_COT_API_KEY,
        model=LLM_COT_MODEL,
    )
    app.state.embedding = SiliconEmbeddingAgent(
        url=EMBEDDING_API_BASE,
        api_key=EMBEDDING_API_KEY,
        model=EMBEDDING_MODEL,
    )
    yield

app = FastAPI(lifespan=lifespan)

for router in all_routers:
    app.include_router(router)

@app.get("/")
async def root():
    return {"message": "llm is ready"}

