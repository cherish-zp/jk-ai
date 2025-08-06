class Config:
    EMBED_MODEL_PATH = r"/Users/zhangpeng/code_bigmodel/jk-ai/00-model/modelscope/embedding_model/sungw111/text2vec-base-chinese-sentence"
    LLM_MODEL_PATH = r"/Users/zhangpeng/code_bigmodel/jk-ai/00-model/modelscope/Qwen/Qwen1.5-1.8B-Chat"

    DATA_DIR = "/Users/zhangpeng/code_bigmodel/jk-ai/rag_law/data"
    VECTOR_DB_DIR = "/Users/zhangpeng/code_bigmodel/jk-ai/rag_law/chroma_db"
    PERSIST_DIR = "/Users/zhangpeng/code_bigmodel/jk-ai/rag_law/storage"

    COLLECTION_NAME = "chinese_labor_laws"
    TOP_K = 10