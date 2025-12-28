"""
10.3.4 在智能体中使用A2A工具
（1）使用A2ATool包装器
"""

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import A2ATool
from dotenv import load_dotenv

load_dotenv(override=True)
llm = HelloAgentsLLM()

# 假设已经有一个研究员Agent服务运行在 http://localhost:5000

# 创建协调者Agent
coordinator = SimpleAgent(name="协调者", llm=llm)

# 添加A2A工具，连接到研究员Agent
researcher_tool = A2ATool(agent_url="http://localhost:5000")
researcher_tool.auto_expand = True
coordinator.add_tool(researcher_tool)

# 协调者可以调用研究员Agent
# 使用 action="ask" 向 Agent 提问
response = coordinator.run("使用a2a工具，向Agent提问：请研究AI在教育领域的应用")
print(response)

"""
TODO:
- 先启动 agent server: python 09_A2A_Server.py; 再运行本例, 遇到下面错误:
✅ 工具 'a2a' 已注册。
抱歉，我当前无法使用 `a2a` 工具。如果您有其他问题或需要帮助，请告诉我！
✅ 工具 'a2a' 已注册。
对不起，我无法提供完整的回答。在使用a2a工具时，我遇到了一个错误，提示必须指定 action 参数。请确保在调用工具时提供了必要的参数。
✅ 工具 'a2a' 已注册。
我明白了。看来工具 `a2a` 不支持我指定的操作 `research`。让我们尝试一个不同的操作，比如 `query`，来获取关于AI在教育领域的应用信息。
请稍等，我将重新尝试调用工具。
```plaintext
```
"""