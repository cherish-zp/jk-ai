from llama_index.core import SimpleDirectoryReader
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter

input_dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-data/jing_rong_csv"


from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir=input_dir)
documents = reader.load_data(True , 1  )


print(documents)
print("/r/n")


loader = CSVLoader(
    file_path=input_dir+"/jr_sy.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["name", "flag"]  # 指定列名
    },
    source_column="name"  # 指定哪一列作为来源元数据
)
documents = loader.load()
print("\r\n")
print(documents)
