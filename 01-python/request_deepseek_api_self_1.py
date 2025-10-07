import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from  okCheck2 import  get_safety_detector,format_detailed_results,quick_safety_check

###  uvicorn  request_deepseek_api_2:app --host 0.0.0.0 --port 8000 --reload
###   python -m http.server 8000

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

# 创建 FastAPI 应用实例
app = FastAPI(title="DeepSeek API 代理服务")

# 添加 CORS 中间件
# 添加 CORS 中间件 - 允许所有域名跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法，包括 OPTIONS
    allow_headers=["*"],  # 允许所有头
)

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


        is_safe, results = quick_safety_check(request.content)
        if is_safe:
            detailed_report = "扫描通过"
        else:
            detailed_report = format_detailed_results(results)

        # 调用 DeepSeek API
        ##response = client.chat.completions.create(
        ##    model="deepseek-chat",
        ##    messages=[
        ##        {"role": "system", "content":"" },  # request.system_prompt
        ##        {"role": "user", "content": request.content},
        ##    ],
        ##    stream=False
        ##)
##
        ### 获取模型回复
        ##assistant_content = response.choices[0].message.content

        assistant_content = detailed_report
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


# 专门处理 OPTIONS 请求（可选，CORS 中间件通常会自动处理）
@app.options("/chat")
async def options_chat():
    return {"message": "OK"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)