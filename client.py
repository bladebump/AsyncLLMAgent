from core.llms.openai_llm import OpenAICoT
from core.embeddings.silicon_agent import SiliconEmbeddingAgent
# deepseek_v3_cot = OpenAICoT(
#     api_base="https://api.siliconflow.cn/v1",
#     api_key="sk-ikrvbgsuezjiomgtczsggqgwwuexjvaxksdabgkcknejbklg",
#     model="Pro/deepseek-ai/DeepSeek-V3",
# )
# deepseek_r1_cot = OpenAICoT(
#     api_base="https://api.siliconflow.cn/v1",
#     api_key="sk-ikrvbgsuezjiomgtczsggqgwwuexjvaxksdabgkcknejbklg",
#     model="Pro/deepseek-ai/DeepSeek-R1",
# )

deepseek_v3_cot = OpenAICoT(
    api_base="https://api.deepseek.com",
    api_key="sk-d6aaf51c1f954197b377b023551c700a",
    model="deepseek-chat",
)
deepseek_r1_cot = OpenAICoT(
    api_base="https://api.deepseek.com",
    api_key="sk-d6aaf51c1f954197b377b023551c700a",
    model="deepseek-reasoner",
)

embedding_agent = SiliconEmbeddingAgent(
    url="https://api.siliconflow.cn/v1/embeddings",
    api_key="sk-ikrvbgsuezjiomgtczsggqgwwuexjvaxksdabgkcknejbklg",
    model="Pro/BAAI/bge-m3",
)
