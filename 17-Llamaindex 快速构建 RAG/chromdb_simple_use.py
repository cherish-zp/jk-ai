import chromadb
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = SentenceTransformer(model_path, device=device)
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input, convert_to_numpy=True).tolist()

# 创建/加载集合（含自定义嵌入函数）
embed_model = SentenceTransformerEmbeddingFunction(
    model_path="/Users/zhangpeng/code_bigmodel/jk-ai/00-model/modelscope/sungw111/text2vec-base-chinese-sentence",
    device="mps"  # 无 GPU 改为 "cpu"
)

# 创建客户端和集合
client = chromadb.Client()
collection = client.create_collection("my_knowledge_base",metadata={"hnsw:space": "cosine"},embedding_function=embed_model)

# 添加文档
collection.add(
    documents=["RAG是一种检索增强生成技术", "向量数据库存储文档的嵌入表示","三英战吕布"],
    metadatas=[{"source": "tech_doc"}, {"source": "tutorial"}, {"source": "tutorial1"}],
    ids=["doc1", "doc2","doc3"]
)

# 查询相似文档
results = collection.query(
    query_texts=["什么是RAG技术？"],
    n_results=3
)
print("---")
print(results)
print("---")


collection.update(
    ids=["doc1"],  # 使用已存在的ID
    documents=["更新后的RAG技术内容"]
)

# 查看更新后的内容 - 方法1：使用get()获取特定ID的内容
updated_docs = collection.get(ids=["doc1"])
print("更新后的文档内容：", updated_docs["documents"])

# 查看更新后的内容 - 方法2：查询所有文档
all_docs = collection.get()
print("集合中所有文档：", all_docs["documents"])

#删除内容
collection.delete(ids=["doc1"])

# 查看更新后的内容 - 方法2：查询所有文档
all_docs = collection.get()
print("集合中所有文档：", all_docs["documents"])

#统计条目
print(collection.count())