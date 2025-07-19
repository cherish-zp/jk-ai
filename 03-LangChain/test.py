from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
import concurrent.futures
import pathlib

class BgeM3Embeddings(Embeddings):
    def __init__(self, model_path: str, device: str = "cpu"):
        # Check if device is available
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        elif device == "mps" and not hasattr(torch.backends, "mps") and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            device = "cpu"
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.device = torch.device(device)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def _average_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ensure texts is List[str] type
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All elements in texts must be of type str")

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._average_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def main():

    try:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(env_path, encoding="utf-8")

        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not found in environment variables")
    except Exception as e:
        print(f"Error loading environment variables: {str(e)}")


    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        return


    project_root = pathlib.Path(__file__).parent.parent
    data_path = project_root / "00-data" / "Weibo" / "万条金融标准术语.csv"
    model_path = project_root / "00-model" / "BAAI" / "bge-m3" / "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    try:
        print(f"Loading data from {data_path}")
        loader = CSVLoader(str(data_path))
        pages = loader.load_and_split()
        print(f"Loaded {len(pages)} documents")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    try:
        print(f"Initializing embeddings with device: {device}")
        embeddings = BgeM3Embeddings(model_path=str(model_path), device=device)
    except Exception as e:
        print(f"Error initializing embeddings: {str(e)}")
        return

    try:
        print("Creating vector database...")
        # Simply use FAISS.from_documents for simplicity
        db = FAISS.from_documents(pages, embeddings)
        print("Vector database created successfully")
    except Exception as e:
        print(f"Error creating vector database: {str(e)}")
        return

    retriever = db.as_retriever(search_kwargs={"k": 3})

    template = """根据以下上下文信息回答问题: 根据我的输入来选出最中的一条结果:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test query
    query = "Probabil"
    try:
        print(f"\n执行查询: '{query}'")
        print("检索相关文档...")
        docs = retriever.get_relevant_documents(query)
        print(f"找到 {len(docs)} 个相关文档")
        context_str = "\n\n".join([d.page_content for d in docs])

        print(context_str)
        

        result = rag_chain.invoke(query)

    except Exception as e:
        print(f"查询失败: {str(e)}")


if __name__ == "__main__":
    main()