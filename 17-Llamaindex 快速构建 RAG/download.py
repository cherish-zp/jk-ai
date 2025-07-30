#模型下载
from modelscope import snapshot_download

dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-model/modelscope" ;
model_name = "sungw111/text2vec-base-chinese-sentence"
model_dir = snapshot_download(model_name , cache_dir=dir)


print("模型下载完成")