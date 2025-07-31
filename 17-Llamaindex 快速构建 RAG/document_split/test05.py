from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import TextNode
import json
import torch

data_dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-data/rag/";
input_dir="/Users/zhangpeng/code_bigmodel/jk-ai/00-model/modelscope/"
model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

qwen_model_name="Qwen/Qwen1.5-1.8B-Chat"
# 1. 初始化本地模型
def setup_local_models():
    # 设置本地embedding模型
    embed_model = HuggingFaceEmbedding(
        model_name=input_dir + model_name,
        device="mps" if torch.mps.is_available() else "cpu"
    )
    
    # 设置本地LLM模型
    llm = HuggingFaceLLM(
        model_name  = input_dir + qwen_model_name,
        tokenizer_name  =input_dir + qwen_model_name,
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True},
        device_map="auto",
        generate_kwargs={"temperature": 0.3, "do_sample": True}  # 修改为do_sample=True避免警告
    )
    
    # 全局设置
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 512

# 2. 加载数据并处理格式
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = []

    query = "如何预防机器学习模型过拟合？" ;
    positive_passages =  "正则化方法通过添加L1/L2惩罚项控制模型复杂度..." ;
    text = f"查询: {query}\n相关文档: {positive_passages}"
    print(text)
    node = TextNode(text=text)
    nodes.append(node)

    query = "如何预防机器学习模型过拟合？" ;
    positive_passages =  "交叉验证将数据划分为训练集和验证集..." ;
    text = f"查询: {query}\n相关文档: {positive_passages}"
    print(text)
    node = TextNode(text=text)
    nodes.append(node)

    query = "如何预防机器学习模型过拟合？" ;
    positive_passages =  "早停法（Early Stopping）监控验证集损失..." ;
    text = f"查询: {query}\n相关文档: {positive_passages}"
    print(text)
    node = TextNode(text=text)
    nodes.append(node)


    
    return nodes

# 3. 初始化本地模型
setup_local_models()

# 4. 加载数据
data_path = data_dir + "qa_pairs.json"

nodes = load_data(data_path)

# 5. 示例查询
query = "如何预防机器学习模型过拟合？"

# 案例1：向量检索（使用本地embedding模型）
vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=3)
print("向量检索结果：", [node.text[:50] + "..." for node in vector_retriever.retrieve(query)])

# 案例2：关键词检索（不使用bm25模式）
from llama_index.core import KeywordTableIndex
keyword_index = KeywordTableIndex(nodes)
keyword_retriever = keyword_index.as_retriever(similarity_top_k=3)  # 使用默认模式
print("关键词检索结果：", [node.text[:50] + "..." for node in keyword_retriever.retrieve(query)])

# 案例3：查询引擎（使用本地LLM生成回答）
query_engine = keyword_retriever.as_query_engine()
response = query_engine.query(query)
print("LLM生成回答：", response)