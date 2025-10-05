import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI


## pip install fastapi uvicorn openai pydantic


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("deepseek_api")

# 创建 FastAPI 应用实例 - 这个必须命名为 app
app = FastAPI(title="DeepSeek API 代理服务")

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-23482e4f07ce4f89b8e15c3e5a6b96d3",
    base_url="https://api.deepseek.com"
)


# 请求模型
class ChatRequest(BaseModel):
    content: str
    system_prompt: str = "You are a helpful assistant"


# 响应模型
class ChatResponse(BaseModel):
    user_content: str
    assistant_content: str
    status: str


@app.post("/chat", response_model=ChatResponse)
async def chat_with_deepseek(request: ChatRequest):
    """
    与 DeepSeek 模型对话的接口
    """
    try:
        # 记录用户请求内容
        logger.info(f"用户请求内容: {request.content}")
        ## logger.info(f"系统提示词: {request.system_prompt}")

        # 调用 DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.content},
            ],
            stream=False
        )

        # 获取模型回复
        assistant_content = response.choices[0].message.content

        # 记录模型回复内容
        logger.info(f"模型回复内容: {assistant_content}")

        # 返回响应
        return ChatResponse(
            user_content=request.content,
            assistant_content=assistant_content,
            status="success"
        )

    except Exception as e:
        # 记录错误日志
        logger.error(f"API 调用失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API 调用失败: {str(e)}")


@app.get("/")
async def root():
    return {"message": "DeepSeek API 代理服务正在运行"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# 如果直接运行此文件
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)