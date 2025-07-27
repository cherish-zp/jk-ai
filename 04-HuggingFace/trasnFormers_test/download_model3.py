from transformers import AutoModelForCausalLM,AutoTokenizer
model_name = "shibing624/text2vec-base-chinese"
cache_dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-model/shibing624/text2vec-base-chinese"

AutoModelForCausalLM.from_pretrained(model_name,cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)

print(f"模型分词器已下载到：{cache_dir}")


