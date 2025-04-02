# 模型参数
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 4096
LLM_TIMEOUT = 600

# LLM_settings
LLM_MODEL = "deepseek-chat"
LLM_API_KEY = "sk-d6aaf51c1f954197b377b023551c700a"
LLM_API_BASE = "https://api.deepseek.com"
# LLM_MODEL="Pro/deepseek-ai/DeepSeek-V3"
# LLM_API_KEY="sk-ikrvbgsuezjiomgtczsggqgwwuexjvaxksdabgkcknejbklg"
# LLM_API_BASE="https://api.siliconflow.cn/v1"

# LLM_COT_settings
LLM_COT_MODEL = "deepseek-reasoner"
LLM_COT_API_KEY = "sk-d6aaf51c1f954197b377b023551c700a"
LLM_COT_API_BASE = "https://api.deepseek.com"
# LLM_COT_MODEL="Pro/deepseek-ai/DeepSeek-R1"
# LLM_COT_API_KEY="sk-ikrvbgsuezjiomgtczsggqgwwuexjvaxksdabgkcknejbklg"
# LLM_COT_API_BASE="https://api.siliconflow.cn/v1"

# Embedding_settings
EMBEDDING_MODEL = "Pro/BAAI/bge-m3"
EMBEDDING_API_KEY = "sk-ikrvbgsuezjiomgtczsggqgwwuexjvaxksdabgkcknejbklg"
EMBEDDING_API_BASE = "https://api.siliconflow.cn/v1/embeddings"

# Milvus_settings
MILVUS_URI = "http://192.170.1.190:19530"
MILVUS_USERNAME = ""
MILVUS_PASSWORD = ""
MILVUS_DENSE_VECTOR_DIM = 1024
MILVUS_USE_SPARSE_VECTOR = False

# 法律相关问答
LAW_QA_THOULD = 0.5

# chunck settings
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# RAG 参数
RAG_TOP_K = 5