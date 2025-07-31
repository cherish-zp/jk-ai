from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
import os


# 2. 加载文档
documents = SimpleDirectoryReader(input_files=["/home/cw/projects/demo_20/data/test.txt"]).load_data()

# # 3. 筛选Markdown文档
# md_docs = [d for d in documents if d.metadata["file_path"].endswith(".md")]

# 4. 初始化模型和解析器
embed_model = HuggingFaceEmbedding(
    #指定了一个预训练的sentence-transformer模型的路径
    model_name="/home/cw/llms/embedding_model/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

semantic_parser = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=90,
    embed_model=embed_model
)

# 5. 执行语义分割
semantic_nodes = semantic_parser.get_nodes_from_documents(documents)

# 6. 打印结果
print(f"语义分割节点数: {len(semantic_nodes)}")
for i, node in enumerate(semantic_nodes[:2]):  # 只打印前两个节点
    print(f"\n节点{i+1}:\n{node.text}")
    print("-"*50)