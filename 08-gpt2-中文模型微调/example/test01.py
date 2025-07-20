#中文白话文文章生成
# 使用 AutoTokenizer , AutoModel 和 GPT2LMHeadModel,BertTokenizer 本质上没有区别
# from transformers import AutoTokenizer , AutoModel

# 这些开放出来预训练号的模型称为预训练模型 ， 预训练模型不是用来给别人使用的，而别用来当做基座的

from transformers import GPT2LMHeadModel,BertTokenizer,TextGenerationPipeline

# 加载模型和分词器
moder_dir = r"/Users/zhangpeng/code_bigmodel/jk-ai/00-model/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3" ;
model = GPT2LMHeadModel.from_pretrained(moder_dir)
tokenizer = BertTokenizer.from_pretrained(moder_dir)
print(model)

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model,tokenizer,device="mps")

#使用text_generator生成文本
#do_sample是否进行随机采样。为True时，每次生成的结果都不一样；为False时，每次生成的结果都是相同的。
for i in range(3):
    print(text_generator("这是很久之前的事情了,", max_length=100, do_sample=True))