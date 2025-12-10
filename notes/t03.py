# 8.1.2 模型本身不会自动“记住”上一次对话的内容。
from dotenv import load_dotenv
from hello_agents import SimpleAgent, HelloAgentsLLM

load_dotenv(override=True)

llm = HelloAgentsLLM()
print(f"model = {llm.model}, base_url = {llm.base_url}")

agent = SimpleAgent(name="学习助手", llm=llm)

# 第一次对话
response1 = agent.run("我叫张三，正在学习Python，目前掌握了基础语法")
print(response1)  # "很好！Python基础语法是编程的重要基础..."
 
# 第二次对话（新的会话）
response2 = agent.run("你还记得我的学习进度吗？")
print(response2)  # "抱歉，我不知道您的学习进度..."
