from llama_index.core import SimpleDirectoryReader
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter

input_dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-data/jing_rong_csv"


from llama_index.core import SimpleDirectoryReader

# 加载CSV
loader = CSVLoader(file_path=input_dir+"/jr_sy.csv")
docs = loader.load()

# 文本分割
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(docs)
