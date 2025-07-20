#模型下载
from modelscope import snapshot_download

dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-model/modelscope" ;
model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct' , cache_dir=dir)


print("模型下载完成")