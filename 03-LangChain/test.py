from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings


class BgeM3Embeddings(Embeddings):
    def __init__(self, model_path: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

    def _average_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._average_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# 保证操作系统的环境变量里面配置好了OPENAI_API_KEY, OPENAI_BASE_URL
load_dotenv(".env", encoding="utf-8")  # 指定文件编码

key  = os.getenv("OPENAI_API_KEY")
url = os.getenv("OPENAI_API_BASE_URL")


llm = ChatOpenAI(model="gpt-3.5-turbo")  # 默认是gpt-3.5-turbo
## response = llm.invoke("你是谁")
## print(response.content)


loader = CSVLoader(r"/Users/zhangpeng/code_bigmodel/jk-ai/00-data/Weibo/万条金融标准术语.csv")

pages = loader.load_and_split()


# 替换为你的本地模型路径
model_path = "/Users/zhangpeng/code_bigmodel/jk-ai/00-model/BAAI/bge-m3/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
# embeddings = BgeM3Embeddings(model_path=model_path, device="mps") # 可改为 "cuda" 如果有 GPU

##  /Users/zhangpeng/code_bigmodel/jk-ai/00-model/BAAI/bge-m3/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
## db = FAISS.from_documents(pages, embeddings)

db = Chroma.from_documents(pages, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})
query = "Probabil" ;
doc = retriever.invoke(query)
context_str = "\n\n".join([d.page_content for d in doc])  # 用两个换行符分隔不同文档
print(context_str)
# Prompt模板
template = """根据以下上下文信息回答问题: 根据我的输入来选出最中的一条结果:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Chain
rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)) , "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 测试查询
try:
    result = rag_chain.invoke(query)
    print(f"\n最终结果: {result}")
except Exception as e:
    print(f"查询失败: {str(e)}")