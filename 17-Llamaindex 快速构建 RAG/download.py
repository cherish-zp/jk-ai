#模型下载
from modelscope import snapshot_download

dir = "/Users/zhangpeng/code_bigmodel/jk-ai/00-model/modelscope" ;
model_dir = snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' , cache_dir=dir)


print("模型下载完成")