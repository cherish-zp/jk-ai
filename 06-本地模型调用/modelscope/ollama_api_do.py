#使用openai的API风格调用ollama
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1/",api_key="suibianxie")

chat_completion = client.chat.completions.create(
    messages=[{"role":"user","content":"你好，请介绍下你自己。"}],model="qwen2.5:0.5b"
)
print(chat_completion.choices[0])