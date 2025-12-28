### chapter 8

import os

from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool, RAGTool
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

# 8.1.2 模型本身不会自动“记住”上一次对话的内容。
def t01():
    llm = HelloAgentsLLM()
    print(f"model = {llm.model}, base_url = {llm.base_url}")

    agent = SimpleAgent(name="学习助手", llm=llm)

    # 第一次对话
    response1 = agent.run("我叫张三，正在学习Python，目前掌握了基础语法")
    print(response1)  # "很好！Python基础语法是编程的重要基础..."
    
    # 第二次对话（新的会话）
    response2 = agent.run("你还记得我的学习进度吗？")
    print(response2)  # "抱歉，我不知道您的学习进度..."

# 8.1.4 本章学习目标与快速体验
# 没有具体添加搜索合并插入记忆，还无法如期工作
# 可以用来检查系统各个组件是否正确配置!
# llm: http://192.168.18.77:8080/v1 - Qwen/Qwen2.5-Coder-7B-Instruct
# RAG: http://192.168.18.88:6333
# neo4j: neo4j://192.168.18.88:7687
def t02():
    # 创建LLM实例
    llm = HelloAgentsLLM()
    # llm = HelloAgentsLLM(provider="custom")
    print(f"model = {llm.model}, base_url = {llm.base_url}")

    # 创建Agent
    agent = SimpleAgent(
        name="智能助手",
        llm=llm,
        system_prompt="你是一个有记忆和知识检索能力的AI助手"
    )

    # 创建工具注册表
    tool_registry = ToolRegistry()

    # 添加记忆工具
    memory_tool = MemoryTool(user_id="user123")
    tool_registry.register_tool(memory_tool)

    # 添加RAG工具
    rag_tool = RAGTool(knowledge_base_path="./knowledge_base")
    tool_registry.register_tool(rag_tool)

    # 为Agent配置工具
    agent.tool_registry = tool_registry

    # 开始对话
    response = agent.run("你好！请记住我叫张三，我是一名Python开发者")
    print(response)
    # 你好，张三！很高兴认识你。作为你的AI助手，我会尽力帮助你解答问题和提供支持。如果你有任何关于Python的问题或需要帮助的地方，请随时告诉我！

    # 没有搜索插入记忆, 还无法如期工作!
    # response2 = agent.run("你还记得我的学习进度吗？")
    # print(response2)

# 8.2.2 快速体验：30秒上手记忆功能
def t03():
    # 创建具有记忆能力的Agent
    llm = HelloAgentsLLM()
    agent = SimpleAgent(name="记忆助手", llm=llm)

    # 创建记忆工具
    memory_tool = MemoryTool(user_id="user123")
    tool_registry = ToolRegistry()
    tool_registry.register_tool(memory_tool)
    agent.tool_registry = tool_registry
    
    # 体验记忆功能
    print("=== 添加多个记忆 ===")

    # 添加第一个记忆
    result1 = memory_tool.execute("add", content="用户张三是一名Python开发者，专注于机器学习和数据分析", memory_type="semantic", importance=0.8)
    print(f"记忆1: {result1}")

    # 添加第二个记忆
    result2 = memory_tool.execute("add", content="李四是前端工程师，擅长React和Vue.js开发", memory_type="semantic", importance=0.7)
    print(f"记忆2: {result2}")

    # 添加第三个记忆
    result3 = memory_tool.execute("add", content="王五是产品经理，负责用户体验设计和需求分析", memory_type="semantic", importance=0.6)
    print(f"记忆3: {result3}")

    print("\n=== 搜索特定记忆 ===")
    # 搜索前端相关的记忆
    print("🔍 搜索 '前端工程师':")
    result = memory_tool.execute("search", query="前端工程师", limit=3)
    print(result)

    print("\n=== 记忆摘要 ===")
    result = memory_tool.execute("summary")
    print(result)

# 8.3.3 快速体验：30秒上手RAG功能
def t05():
    # 创建具有RAG能力的Agent
    llm = HelloAgentsLLM()
    agent = SimpleAgent(name="知识助手", llm=llm)

    # 创建RAG工具
    rag_tool = RAGTool(
        knowledge_base_path="./knowledge_base",
        collection_name="test_collection",
        rag_namespace="test"
    )

    tool_registry = ToolRegistry()
    tool_registry.register_tool(rag_tool)
    agent.tool_registry = tool_registry

    # 体验RAG功能
    # 添加第一个知识
    result1 = rag_tool.execute("add_text", 
        text="Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。Python的设计哲学强调代码的可读性和简洁的语法。",
        document_id="python_intro")
    print(f"知识1: {result1}")

    # 添加第二个知识  
    result2 = rag_tool.execute("add_text",
        text="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。主要包括监督学习、无监督学习和强化学习三种类型。",
        document_id="ml_basics")
    print(f"知识2: {result2}")

    # 添加第三个知识
    result3 = rag_tool.execute("add_text",
        text="RAG（检索增强生成）是一种结合信息检索和文本生成的AI技术。它通过检索相关知识来增强大语言模型的生成能力。",
        document_id="rag_concept")
    print(f"知识3: {result3}")


    print("\n=== 搜索知识 ===")
    result = rag_tool.execute("search",
        query="Python编程语言的历史",
        limit=3,
        min_score=0.5
    )
    print(result)

    print("\n=== 知识库统计 ===")
    result = rag_tool.execute("stats")
    print(result)

if __name__ == "__main__":
    # t01()
    # t02()
    t03()
    # t05()
    pass