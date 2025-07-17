#将模型下载到本地调用
from transformers import AutoModelForCausalLM,AutoTokenizer

#将模型和分词工具下载到本地，并指定保存路径
model_name = "BAAI/bge-m3"
cache_dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-model/BAAI/bge-m3"

#下载模型
AutoModelForCausalLM.from_pretrained(model_name,cache_dir=cache_dir)
#下载分词工具
AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)

print(f"模型分词器已下载到：{cache_dir}")


