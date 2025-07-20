# 使用 transform 加载 qwen模型
from modelscope import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps"

model_dir = r"/Users/zhangpeng/code_bigmodel/jk-ai/00-model/modelscope/Qwen/Qwen2.5-0.5B-Instruct"


#使用transformer加载模型
model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype="auto",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

#调用模型
#定义提示词
prompt = "你好，请介绍下你自己。"
#将提示词封装为message
message = [{"role":"system","content":"You are a helpful assistant system"},{"role":"user","content":prompt}]
#使用分词器的apply_chat_template()方法将上面定义的消息列表进行转换;tokenize=False表示此时不进行令牌化
text = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)

#将处理后的文本令牌化并转换为模型的输入张量
model_inputs = tokenizer([text],return_tensors="pt").to(DEVICE)

#将数据输入模型得到输出  input_ids 收入文本对应的 字典的索引
response = model.generate(model_inputs.input_ids,max_new_tokens=512)
print(response)

#对输出的内容进行解码还原 skip_special_tokens=True 跳过特殊符号
response = tokenizer.batch_decode(response,skip_special_tokens=True)
print(response)

