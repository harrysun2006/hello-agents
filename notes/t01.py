# 测试 Gemini model 调用
import os
import google.generativeai as genai

from dotenv import load_dotenv
# from google.generativeai import GenerativeModel

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai.configure(transport='grpc')

def list_models():
    for i, m in zip(range(200), genai.list_models()):
        print(f"{m.name}: {m.description}; support = {m.supported_generation_methods}")

def greet():
    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")
    model = genai.GenerativeModel(model_name)
    print(f"model_name = {model_name}")
    response = model.generate_content("用中文介绍一下 Python 的 GIL 是什么")
    print(response.text)

if __name__ == "__main__":
    # list_models()
    greet()
