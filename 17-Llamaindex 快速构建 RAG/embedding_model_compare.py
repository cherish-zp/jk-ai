from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np

# 加载 BGE 中文嵌入模型
model_name = "/home/cw/llms/embedding_model/sungw111/text2vec-base-chinese-sentence"
model_name = "/Users/zhangpeng/code_bigmodel/jk-ai/00-model/modelscope/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 

model_name = "/Users/zhangpeng/code_bigmodel/jk-ai/00-model/modelscope/sungw111/text2vec-base-chinese-sentence" 

embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    device="mps",  # 使用 GPU，如果没有 GPU 改为 "cpu"
    normalize=True,  # 归一化向量，方便计算余弦相似度
)

# 嵌入文档
documents = ["忘记密码如何处理？", "用户账号被锁定"]
doc_embeddings = [embed_model.get_text_embedding(doc) for doc in documents]

# 嵌入查询并计算相似度
query = "密码重置流程"
query_embedding = embed_model.get_text_embedding(query)

# 计算余弦相似度（因为 normalize=True，点积就是余弦相似度）
similarity = np.dot(query_embedding, doc_embeddings[0])
print(f"相似度：{similarity:.4f}")  # 输出示例：0.8512