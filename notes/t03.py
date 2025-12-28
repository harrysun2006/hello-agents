# 测试 https://github.com/jjyaoao/HelloAgents/blob/main/README.md 中的代码
from hello_agents import HelloAgentsLLM, ToolRegistry, search, calculate
from hello_agents import SimpleAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent
from dotenv import load_dotenv

load_dotenv(override=True)

# 创建LLM实例 - 框架自动检测provider
llm = HelloAgentsLLM()

# 或手动指定provider（可选）
# llm = HelloAgentsLLM(provider="modelscope")

# 基本使用
# agent.stream_run 会产生重复chunks, why?
def t00():
    # 创建SimpleAgent
    agent = SimpleAgent(
        name="AI助手",
        llm=llm,
        system_prompt="你是一个有用的AI助手"
    )

    # 开始对话
    response = agent.run("你好！请介绍一下自己")
    print(response)

    # 流式对话
    print("助手: ", end="", flush=True)
    for chunk in agent.stream_run("什么是人工智能？"):
        print(chunk, end="", flush=True)
    print()

    # 检查自动检测结果
    print(f"自动检测的provider: {llm.provider}")

# 1. ReActAgent - 推理与行动结合
# --- 第 1 步 ---
# ⚠️ 警告：未能解析出有效的Action，流程终止。
# ⏰ 已达到最大步数，流程终止。
def t01():
    # 创建工具注册表
    tool_registry = ToolRegistry()
    tool_registry.register_function("search", "网页搜索工具", search)
    tool_registry.register_function("calculate", "数学计算工具", calculate)

    # 创建ReAct Agent
    react_agent = ReActAgent(
        name="研究助手",
        llm=llm,
        tool_registry=tool_registry,
        max_steps=5
    )

    # 执行需要工具的任务
    result = react_agent.run("搜索最新的GPT-4发展情况，并计算其参数量相比GPT-3的增长倍数")

# 2. ReflectionAgent - 自我反思与迭代优化
def t02():
    # 创建Reflection Agent
    reflection_agent = ReflectionAgent(
        name="代码专家",
        llm=llm,
        max_iterations=3
    )

    # 生成并优化代码
    code = reflection_agent.run("编写一个高效的素数筛选算法，要求时间复杂度尽可能低")
    print(f"最终代码:\n{code}")

# 3. PlanAndSolveAgent - 分解规划与逐步执行
# model=Qwen/Qwen2.5-Coder-7B-Instruct
# 结果不稳定: 77.4, 40.36, 108.145, 229.7, ...
# 1) 加入temperature=0.0 之后稳定=77.4
# 答案错误! 漏了第一年利润!
# 2) prompt 加入"列计划时请分别计算逐年的营收，成本，利润。" 答案正确!
# 107.4万!!
# 直接在chat-ui 提问可以得出正确答案!
def t03():
    # 创建Plan and Solve Agent
    plan_agent = PlanAndSolveAgent(name="问题解决专家", llm=llm)

    # 解决复杂问题
    problem = """
    一家公司第一年营收100万，第二年增长20%，第三年增长15%。
    如果每年的成本是营收的70%，请计算三年的总利润。
    列计划时请分别计算逐年的营收，成本，利润。
    """
    answer = plan_agent.run(problem, temperature=0.0)
    print(answer)

if __name__ == "__main__":
    # t00()
    # t01()
    # t02()
    t03()
    pass
