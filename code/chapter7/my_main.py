# my_main.py
from dotenv import load_dotenv
from hello_agents.core.llm import HelloAgentsLLM
from my_llm import MyLLM # 注意：这里导入我们自己的类

# 加载环境变量
load_dotenv(override=True)

# 实例化我们重写的客户端，并指定provider
# llm = MyLLM(provider="modelscope") 
llm = HelloAgentsLLM(provider="custom", temperature=0.1)

# 准备消息
messages = [
    {"role": "system", "content": "You are a helpful AI assistant"},
    {"role": "user", "content": "你好，请介绍一下你自己。"}
]

# 发起调用，think等方法都已从父类继承，无需重写
# think will respond duplicated characters like: 你好你好！！我我是一个是一个由由...
# response_stream = llm.think(messages)
response_stream = llm.invoke(messages)

# 打印响应
print(f"{llm.model} @ {llm.base_url} Response:")
for chunk in response_stream:
    # chunk 已经是文本片段，可以直接使用
    print(chunk, end="", flush=True)