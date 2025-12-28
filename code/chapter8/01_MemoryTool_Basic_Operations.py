#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码示例 01: MemoryTool基础操作
展示MemoryTool的核心execute方法和基本操作
"""

from datetime import datetime
from typing import List
from hello_agents.tools import MemoryTool
from dotenv import load_dotenv

load_dotenv(override=True)

def memory_tool_execute_demo():
    """MemoryTool execute方法演示"""
    print("🧠 MemoryTool基础操作演示")
    print("=" * 50)
    
    # 初始化MemoryTool
    memory_tool = MemoryTool(
        user_id="demo_user",
        memory_types=["working", "episodic", "semantic", "perceptual"]
    )
    
    print("✅ MemoryTool初始化完成")
    print(f"📋 支持的操作: add, search, summary, stats, update, remove, forget, consolidate, clear_all")
    
    return memory_tool

def add_memory_demo(memory_tool):
    """添加记忆演示 - 模拟人类记忆编码过程"""
    print("\n📝 添加记忆演示")
    print("-" * 30)
    
    # 添加工作记忆
    result = memory_tool.execute(
        "add",
        content="正在学习HelloAgents框架的记忆系统",
        memory_type="working",
        importance=0.7,
        task_type="learning"
    )
    print(f"工作记忆: {result}")
    
    # 添加情景记忆
    result = memory_tool.execute(
        "add",
        content="2024年开始深入研究AI Agent技术",
        memory_type="episodic",
        importance=0.8,
        event_type="milestone",
        location="研发中心"
    )
    print(f"情景记忆: {result}")
    result = memory_tool.execute(
        "add",
        content="2023年开始训练 LLM 模型",
        memory_type="episodic",
        importance=0.6,
        event_type="milestone",
        location="研发中心"
    )
    
    # 添加语义记忆
    result = memory_tool.execute(
        "add",
        content="记忆系统包括工作记忆、情景记忆、语义记忆和感知记忆四种类型",
        memory_type="semantic",
        importance=0.9,
        concept="memory_types",
        domain="cognitive_science"
    )
    print(f"语义记忆: {result}")
    
    # 添加感知记忆
    result = memory_tool.execute(
        "add",
        content="查看了记忆系统的架构图和实现代码",
        memory_type="perceptual",
        importance=0.6,
        modality="document",
        source="technical_documentation"
    )
    print(f"感知记忆: {result}")

def search_memory_demo(memory_tool):
    """搜索记忆演示 - 实现语义理解的检索"""
    print("\n🔍 搜索记忆演示")
    print("-" * 30)
    
    # 基础搜索
    print("基础搜索 - '记忆系统':")
    result = memory_tool.execute("search", query="记忆系统", limit=3)
    print(result)
    """
    1. [语义记忆] 记忆系统包括工作记忆、情景记忆、语义记忆和感知记忆四种类型 (重要性: 0.90)
    2. [情景记忆] 2024年开始深入研究AI Agent技术 (重要性: 0.80)
    3. [工作记忆] 正在学习HelloAgents框架的记忆系统 (重要性: 0.70)
    """
    
    # 按类型搜索
    print("\n按类型搜索 - 语义记忆中的'记忆':")
    result = memory_tool.execute(
        "search", 
        query="记忆", 
        memory_type="semantic", 
        limit=2
    )
    print(result)
    """
    1. [语义记忆] 记忆系统包括工作记忆、情景记忆、语义记忆和感知记忆四种类型 (重要性: 0.90)
    """
    
    # 设置重要性阈值
    print("\n高重要性记忆搜索:")
    result = memory_tool.execute(
        "search", 
        # query="AI Agent", 
        query="航天科技",
        min_importance=0.7, 
        limit=2
    )
    print(result)
    
    print("\n高重要性记忆搜索B:")
    result = memory_tool.execute(
        "search", 
        # query="AI Agent", 
        query="航天科技",
        min_importance=0.7, 
        limit=5
    )
    print(result)
    
    print("\n高重要性记忆搜索C:")
    result = memory_tool.execute(
        "search", 
        # query="AI Agent", 
        query="航天科技",
        min_importance=0.59, 
        limit=6
    )
    print(result)

def memory_summary_demo(memory_tool):
    """记忆摘要演示 - 提供系统全貌"""
    print("\n📋 记忆摘要演示")
    print("-" * 30)
    
    # 获取记忆摘要
    result = memory_tool.execute("summary", limit=5)
    print("记忆摘要:")
    print(result)
    
    # 获取统计信息
    print("\n📊 统计信息:")
    result = memory_tool.execute("stats")
    print(result)

def memory_management_demo(memory_tool):
    """记忆管理演示 - 遗忘和整合"""
    print("\n⚙️ 记忆管理演示")
    print("-" * 30)
    
    # 添加一个低重要性记忆用于遗忘测试
    memory_tool.execute(
        "add",
        content="这是一个临时的测试记忆，重要性很低",
        memory_type="working",
        importance=0.1
    ) # RAM
    
    # 基于重要性的遗忘
    print("基于重要性的遗忘 (阈值=0.2):")
    result = memory_tool.execute(
        "forget",
        strategy="importance_based",
        threshold=0.2
    ) # - "这是一个临时的测试记忆，重要性很低" (0.1)
    print(result)
    
    # 记忆整合 - 将重要的工作记忆转为情景记忆
    print("\n记忆整合 (working → episodic):")
    result = memory_tool.execute(
        "consolidate",
        from_type="working",
        to_type="episodic",
        importance_threshold=0.6
    ) # 1 working 0.7 => 1 episodic 0.77
    print(result)

def main():
    """主函数"""
    print("🚀 MemoryTool基础操作完整演示")
    print("展示记忆系统的核心功能和操作方法")
    print("=" * 60)
    
    try:
        # 1. 初始化MemoryTool
        memory_tool = memory_tool_execute_demo()
        
        # 2. 添加记忆演示
        add_memory_demo(memory_tool)
        
        # 3. 搜索记忆演示
        search_memory_demo(memory_tool)
        
        # 4. 记忆摘要演示
        memory_summary_demo(memory_tool)
        
        # 5. 记忆管理演示
        memory_management_demo(memory_tool)
        
        # memory 前后比较
        result = memory_tool.execute("summary", limit=5)
        print("记忆摘要:")
        print(result)
        print("\n📊 统计信息:")
        result = memory_tool.execute("stats")
        print(result)
    
        print("\n" + "=" * 60)
        print("🎉 MemoryTool基础操作演示完成！")
        print("=" * 60)
        
        print("\n✨ 演示的核心功能:")
        print("1. 🧠 四种记忆类型的添加和管理")
        print("2. 🔍 智能语义搜索和过滤")
        print("3. 📋 记忆摘要和统计分析")
        print("4. ⚙️ 记忆整合和选择性遗忘")
        
        print("\n🎯 设计特点:")
        print("• 统一的execute接口，操作简洁一致")
        print("• 丰富的元数据支持，便于分类和检索")
        print("• 智能的重要性评估和时间衰减机制")
        print("• 模拟人类认知的记忆管理策略")
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
TODO:
- ⚠️ search_memory_demo 
高重要性记忆搜索 behaviour?
limit=3 => limit=2: 结果从3条(0.9, 0.8, 0.6) => 2条(0.9, 0.8);
limit=3 => limit=5: 还是3条
min_importance=0.7 => min_importance=0.5: 还是3条, 重要性0.6的感知记忆也被选中? 
query 被忽略了? working 类型记忆也被忽略了?

基础搜索 - '记忆系统': 返回3条(0.9, 0.8, 0.7)

高重要性记忆搜索:
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.29it/s]
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.21it/s]
INFO:hello_agents.memory.types.semantic:✅ 检索到 1 条相关记忆
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.59it/s]
🔍 找到 3 条相关记忆:
1. [语义记忆] 记忆系统包括工作记忆、情景记忆、语义记忆和感知记忆四种类型 (重要性: 0.90)
2. [情景记忆] 2024年开始深入研究AI Agent技术 (重要性: 0.80)
3. [感知记忆] 查看了记忆系统的架构图和实现代码 (重要性: 0.60)

- ✅️ 结果验证
📋 记忆类型分布:
  • 工作记忆: 1 条 (平均重要性: 0.70)
  • 情景记忆: 1 条 (平均重要性: 0.80)
  • 语义记忆: 1 条 (平均重要性: 0.90)
  • 感知记忆: 1 条 (平均重要性: 0.60)

⭐ 重要记忆 (前3条):
  1. 记忆系统包括工作记忆、情景记忆、语义记忆和感知记忆四种类型 (重要性: 0.90)
  2. 2024年开始深入研究AI Agent技术 (重要性: 0.80)
  3. 查看了记忆系统的架构图和实现代码 (重要性: 0.60)
===>
📋 记忆类型分布:
  • 工作记忆: 0 条 (平均重要性: 0.00)
  • 情景记忆: 2 条 (平均重要性: 0.79)
  • 语义记忆: 1 条 (平均重要性: 0.90)
  • 感知记忆: 1 条 (平均重要性: 0.60)

⭐ 重要记忆 (前4条):
  1. 记忆系统包括工作记忆、情景记忆、语义记忆和感知记忆四种类型 (重要性: 0.90)
  2. 2024年开始深入研究AI Agent技术 (重要性: 0.80)
  3. 正在学习HelloAgents框架的记忆系统 (重要性: 0.77)
  4. 查看了记忆系统的架构图和实现代码 (重要性: 0.60)

正在学习HelloAgents框架的记忆系统 (working/RAM, 0.7)
==>
正在学习HelloAgents框架的记忆系统 (episodic/sqlite, 0.77=0.7*1.1)

"""