#模型下载
from modelscope import snapshot_download

dir = "/00-model/modelscope";
# model_name = "sungw111/text2vec-base-chinese-sentence"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_dir = snapshot_download(model_name , cache_dir=dir)


print("模型下载完成")