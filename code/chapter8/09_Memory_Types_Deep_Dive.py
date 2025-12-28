#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码示例 09: 四种记忆类型深度解析
详细展示WorkingMemory、EpisodicMemory、SemanticMemory、PerceptualMemory的实现特点
"""

import logging
# logging.basicConfig(level=logging.DEBUG)
logging.getLogger("hello_agents").setLevel(logging.DEBUG)

import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from hello_agents.tools import MemoryTool
from dotenv import load_dotenv

load_dotenv(override=True)

class MemoryTypesDeepDive:
    """四种记忆类型深度解析演示类"""
    
    def __init__(self):
        self.setup_memory_systems()
    
    def setup_memory_systems(self):
        """设置不同的记忆系统"""
        print("🧠 四种记忆类型深度解析")
        print("=" * 60)
        
        # 创建专门的记忆工具实例
        self.working_memory_tool = MemoryTool(
            user_id="working_memory_user",
            memory_types=["working"]
        )
        
        self.episodic_memory_tool = MemoryTool(
            user_id="episodic_memory_user", 
            memory_types=["episodic"]
        )
        
        self.semantic_memory_tool = MemoryTool(
            user_id="semantic_memory_user",
            memory_types=["semantic"]
        )
        
        self.perceptual_memory_tool = MemoryTool(
            user_id="perceptual_memory_user",
            memory_types=["perceptual"]
        )
        
        print("✅ 四种记忆系统初始化完成")
    
    def demonstrate_working_memory(self):
        """演示工作记忆的特点"""
        print("\n💭 工作记忆 (Working Memory) 深度解析")
        print("-" * 60)
        
        print("🔍 工作记忆特点:")
        print("• ⚡ 访问速度极快（纯内存存储）")
        print("• 📏 容量有限（默认50条记忆）")
        print("• ⏰ 自动过期（TTL机制）")
        print("• 🔄 适合临时信息存储")
        
        # 演示容量限制
        print(f"\n1. 容量限制演示:")
        print("添加大量临时记忆，观察容量管理...")
        
        for i in range(8):
            content = f"临时工作记忆 {i+1}: 当前正在处理任务步骤 {i+1}"
            result = self.working_memory_tool.execute("add",
                                                    content=content,
                                                    memory_type="working",
                                                    importance=0.3 + (i * 0.1),
                                                    task_step=i+1)
            print(f"  添加记忆 {i+1}: {result}")
        
        # 检查当前状态
        stats = self.working_memory_tool.execute("stats")
        print(f"\n当前工作记忆状态: {stats}")
        
        # 演示TTL机制
        print(f"\n2. TTL（生存时间）机制演示:")
        
        # 添加一些带时间戳的记忆
        current_time = datetime.now()
        
        # 模拟不同时间的记忆
        time_memories = [
            ("刚刚的想法", 0, 0.8),
            ("5分钟前的任务", 5, 0.6),
            ("10分钟前的提醒", 10, 0.4),
            ("很久以前的笔记", 30, 0.2)
        ]
        
        for content, minutes_ago, importance in time_memories:
            # 这里我们模拟时间差异
            result = self.working_memory_tool.execute("add",
                                                    content=content,
                                                    memory_type="working",
                                                    importance=importance,
                                                    simulated_age_minutes=minutes_ago)
            print(f"  添加记忆: {content} (模拟 {minutes_ago} 分钟前)")
        
        # 演示快速检索
        print(f"\n3. 快速检索演示:")
        
        search_queries = ["任务", "想法", "提醒"]
        
        for query in search_queries:
            start_time = time.time()
            results = self.working_memory_tool.execute("search",
                                                     query=query,
                                                     memory_type="working",
                                                     limit=20)
            search_time = time.time() - start_time
            print(f"  查询 '{query}': {search_time:.4f}秒")
            print(f"  结果: {results[:600]}...")
        
        # 演示自动清理
        print(f"\n4. 自动清理机制:")
        
        # 获取清理前的统计
        before_stats = self.working_memory_tool.execute("stats")
        before_summary = self.working_memory_tool.execute("summary")
        print(f"清理前: {before_stats}\n{before_summary}")
        
        # 触发清理（通过遗忘低重要性记忆）
        forget_result = self.working_memory_tool.execute("forget",
                                                       strategy="importance_based",
                                                       threshold=0.6)
        print(f"清理结果: {forget_result}")
        
        # 获取清理后的统计
        after_stats = self.working_memory_tool.execute("stats")
        after_summary = self.working_memory_tool.execute("summary")
        print(f"清理后: {after_stats}\n{after_summary}")
    
    def demonstrate_episodic_memory(self):
        """演示情景记忆的特点"""
        print("\n📖 情景记忆 (Episodic Memory) 深度解析")
        print("-" * 60)
        
        print("🔍 情景记忆特点:")
        print("• 📅 完整的时间序列记录")
        print("• 🎭 丰富的上下文信息")
        print("• 🔗 支持记忆链条构建")
        print("• 💾 持久化存储")
        
        # 演示完整事件记录
        print(f"\n1. 完整事件记录演示:")
        
        # 模拟一个完整的学习会话
        learning_session = [
            {
                "content": "开始学习Python机器学习",
                "context": "学习开始",
                "location": "家里书房",
                "mood": "专注",
                "importance": 0.7
            },
            {
                "content": "学习了线性回归的数学原理",
                "context": "理论学习",
                "chapter": "第3章",
                "difficulty": "中等",
                "importance": 0.8
            },
            {
                "content": "实现了第一个线性回归模型",
                "context": "实践编程",
                "code_lines": 45,
                "bugs_fixed": 2,
                "importance": 0.9
            },
            {
                "content": "完成了课后练习题",
                "context": "练习巩固",
                "exercises_completed": 5,
                "accuracy": 0.8,
                "importance": 0.6
            },
            {
                "content": "总结今天的学习收获",
                "context": "学习总结",
                "key_concepts": ["线性回归", "梯度下降", "损失函数"],
                "importance": 0.8
            }
        ]
        
        session_id = f"learning_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for i, event in enumerate(learning_session):
            result = self.episodic_memory_tool.execute("add",
                                                     content=event["content"],
                                                     memory_type="episodic",
                                                     importance=event["importance"],
                                                     session_id=session_id,
                                                     sequence_number=i+1,
                                                     **{k: v for k, v in event.items() if k not in ["content", "importance"]})
            print(f"  事件 {i+1}: {result}")
        
        # 演示时间序列检索
        print(f"\n2. 时间序列检索演示:")
        
        # 按时间顺序检索
        timeline_search = self.episodic_memory_tool.execute("search",
                                                        query="学习",
                                                        memory_type="episodic",
                                                        limit=10)
        print(f"学习时间线: {timeline_search}")
        
        # 按会话检索
        session_search = self.episodic_memory_tool.execute("search",
                                                        # query="线性回归",
                                                        query="乌兰巴托的夜",
                                                        memory_type="episodic",
                                                        limit=10)
        print(f"会话内容: {session_search}")
        
        # 演示上下文丰富性
        print(f"\n3. 上下文信息演示:")
        
        # 添加带有丰富上下文的记忆
        rich_context_memory = {
            "content": "参加了AI技术分享会",
            "event_type": "conference",
            "location": "北京国际会议中心",
            "speakers": ["张教授", "李博士", "王工程师"],
            "topics": ["深度学习", "自然语言处理", "计算机视觉"],
            "attendees_count": 200,
            "duration_hours": 6,
            "weather": "晴朗",
            "transportation": "地铁",
            "networking_contacts": 3,
            "key_insights": ["Transformer架构的演进", "多模态学习的前景"],
            "follow_up_actions": ["阅读推荐论文", "尝试新框架"],
            "satisfaction_rating": 9
        }
        
        context_result = self.episodic_memory_tool.execute("add",
                                                         content=rich_context_memory["content"],
                                                         memory_type="episodic",
                                                         importance=0.9,
                                                         **{k: v for k, v in rich_context_memory.items() if k != "content"})
        print(f"丰富上下文记忆: {context_result}")
        
        # 演示记忆链条
        print(f"\n4. 记忆链条构建:")
        
        # 创建相关联的记忆序列
        memory_chain = [
            ("看到一篇关于GPT的论文", "trigger", None),
            ("决定深入研究Transformer架构", "decision", "trigger"),
            ("下载并阅读Attention is All You Need论文", "action", "decision"),
            ("实现了简化版的自注意力机制", "implementation", "action"),
            ("在项目中应用了学到的知识", "application", "implementation")
        ]
        
        chain_memories = {}
        for content, chain_type, parent_type in memory_chain:
            parent_id = chain_memories.get(parent_type) if parent_type else None
            
            result = self.episodic_memory_tool.execute("add",
                                                     content=content,
                                                     memory_type="episodic",
                                                     importance=0.7,
                                                     chain_type=chain_type,
                                                     parent_memory=parent_id,
                                                     chain_id="gpt_learning_chain")
            
            # 提取记忆ID（简化处理）
            memory_id = f"{chain_type}_memory"
            chain_memories[chain_type] = memory_id
            print(f"  链条记忆: {content} (类型: {chain_type})")
        
        # 检索整个链条
        chain_search = self.episodic_memory_tool.execute("search",
                                                        query="GPT Transformer",
                                                        memory_type="episodic",
                                                        limit=50)
        print(f"记忆链条检索: {chain_search}")
    
    def demonstrate_semantic_memory(self):
        """演示语义记忆的特点"""
        print("\n🧠 语义记忆 (Semantic Memory) 深度解析")
        print("-" * 60)
        
        print("🔍 语义记忆特点:")
        print("• 🔗 知识图谱结构化存储")
        print("• 🎯 概念和关系的抽象表示")
        print("• 🔍 语义相似度检索")
        print("• 🧮 支持推理和关联")
        
        # 演示概念存储
        print(f"\n1. 概念知识存储演示:")
        
        # 添加不同类型的概念知识
        concepts = [
            {
                "content": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式",
                "concept_type": "definition",
                "domain": "artificial_intelligence",
                "keywords": ["机器学习", "人工智能", "算法", "数据", "模式"],
                "importance": 0.9
            },
            {
                "content": "监督学习使用标记数据训练模型，包括分类和回归两大类任务",
                "concept_type": "category",
                "domain": "machine_learning",
                "parent_concept": "机器学习",
                "subcategories": ["分类", "回归"],
                "importance": 0.8
            },
            {
                "content": "梯度下降是一种优化算法，通过迭代更新参数来最小化损失函数",
                "concept_type": "algorithm",
                "domain": "optimization",
                "mathematical_basis": "微积分",
                "applications": ["神经网络训练", "线性回归"],
                "importance": 0.8
            },
            {
                "content": "过拟合是指模型在训练数据上表现很好，但在新数据上泛化能力差",
                "concept_type": "problem",
                "domain": "machine_learning",
                "causes": ["模型复杂度过高", "训练数据不足"],
                "solutions": ["正则化", "交叉验证", "早停"],
                "importance": 0.7
            }
        ]
        
        for concept in concepts:
            result = self.semantic_memory_tool.execute("add",
                                                     content=concept["content"],
                                                     memory_type="semantic",
                                                     importance=concept["importance"],
                                                     **{k: v for k, v in concept.items() if k not in ["content", "importance"]})
            print(f"  概念存储: {concept['concept_type']} - {result}")
        
        # 演示关系推理
        print(f"\n2. 关系推理演示:")
        
        # 添加关系知识
        relationships = [
            {
                "content": "深度学习是机器学习的子集，使用多层神经网络",
                "relation_type": "is_subset_of",
                "subject": "深度学习",
                "object": "机器学习",
                "strength": 0.9
            },
            {
                "content": "卷积神经网络特别适合处理图像数据",
                "relation_type": "suitable_for",
                "subject": "卷积神经网络",
                "object": "图像处理",
                "strength": 0.8
            },
            {
                "content": "反向传播算法用于训练神经网络",
                "relation_type": "used_for",
                "subject": "反向传播",
                "object": "神经网络训练",
                "strength": 0.9
            }
        ]
        
        for relation in relationships:
            result = self.semantic_memory_tool.execute("add",
                                                     content=relation["content"],
                                                     memory_type="semantic",
                                                     importance=0.8,
                                                     **{k: v for k, v in relation.items() if k != "content"})
            print(f"  关系存储: {relation['relation_type']} - {result}")
        
        # 演示语义检索
        print(f"\n3. 语义相似度检索:")
        
        semantic_queries = [
            "什么是人工智能？",
            "如何防止模型过拟合？",
            "神经网络的训练方法",
            "图像识别技术",
            "太极拳分哪些流派?"
        ]
        
        for query in semantic_queries:
            start_time = time.time()
            results = self.semantic_memory_tool.execute("search",
                                                      query=query,
                                                      memory_type="semantic",
                                                      limit=30)
            search_time = time.time() - start_time
            print(f"  查询: '{query}' ({search_time:.4f}秒)")
            print(f"  结果: {results[:600]}...")
        
        # 演示知识图谱构建
        print(f"\n4. 知识图谱构建:")
        
        # 添加实体和关系
        entities_and_relations = [
            {
                "content": "TensorFlow是Google开发的深度学习框架",
                "entity_type": "framework",
                "developer": "Google",
                "domain": "deep_learning",
                "language": "Python",
                "year": 2015
            },
            {
                "content": "PyTorch是Facebook开发的深度学习框架，以动态图著称",
                "entity_type": "framework", 
                "developer": "Facebook",
                "domain": "deep_learning",
                "feature": "dynamic_graph",
                "language": "Python"
            },
            {
                "content": "BERT是基于Transformer的预训练语言模型",
                "entity_type": "model",
                "architecture": "Transformer",
                "task": "natural_language_processing",
                "training_method": "pre_training"
            }
        ]
        
        for item in entities_and_relations:
            result = self.semantic_memory_tool.execute("add",
                                                     content=item["content"],
                                                     memory_type="semantic",
                                                     importance=0.8,
                                                     **{k: v for k, v in item.items() if k != "content"})
            print(f"  实体关系: {item['entity_type']} - {result}")
        
        # 获取语义记忆统计
        semantic_stats = self.semantic_memory_tool.execute("stats")
        print(f"\n语义记忆统计: {semantic_stats}")
    
    def demonstrate_perceptual_memory(self):
        """演示感知记忆的特点"""
        print("\n👁️ 感知记忆 (Perceptual Memory) 深度解析")
        print("-" * 60)
        
        print("🔍 感知记忆特点:")
        print("• 🎨 多模态数据支持")
        print("• 🔄 跨模态相似性搜索")
        print("• 📊 感知数据的语义理解")
        print("• 🎯 内容生成和检索")
        
        # 演示文本感知记忆
        print(f"\n1. 文本感知记忆:")
        
        text_perceptions = [
            {
                "content": "这是一段优美的诗歌：春江潮水连海平，海上明月共潮生",
                "modality": "text",
                "genre": "poetry",
                "emotion": "peaceful",
                "language": "chinese",
                "aesthetic_value": 0.9
            },
            {
                "content": "技术文档：API接口返回JSON格式数据，包含状态码和响应体",
                "modality": "text",
                "genre": "technical",
                "complexity": "medium",
                "language": "chinese",
                "practical_value": 0.8
            }
        ]
        
        for perception in text_perceptions:
            result = self.perceptual_memory_tool.execute("add",
                                                       content=perception["content"],
                                                       memory_type="perceptual",
                                                       importance=0.7,
                                                       **{k: v for k, v in perception.items() if k != "content"})
            print(f"  文本感知: {perception['genre']} - {result}")
        
        # 演示图像感知记忆（模拟）
        print(f"\n2. 图像感知记忆（模拟）:")
        
        # 模拟图像数据
        image_perceptions = [
            {
                "content": "一张美丽的日落风景照片",
                "modality": "image",
                "file_path": "/simulated/sunset.jpg",
                "scene_type": "landscape",
                "colors": ["orange", "red", "purple"],
                "objects": ["sun", "clouds", "horizon"],
                "mood": "serene",
                "quality": "high"
            },
            {
                "content": "技术架构图展示了微服务系统设计",
                "modality": "image", 
                "file_path": "/simulated/architecture.png",
                "scene_type": "technical",
                "components": ["API Gateway", "Services", "Database"],
                "complexity": "high",
                "purpose": "documentation"
            }
        ]
        
        for perception in image_perceptions:
            result = self.perceptual_memory_tool.execute("add",
                                                       content=perception["content"],
                                                       memory_type="perceptual",
                                                       importance=0.8,
                                                       **{k: v for k, v in perception.items() if k != "content"})
            print(f"  图像感知: {perception['scene_type']} - {result}")
        
        # 演示音频感知记忆（模拟）
        print(f"\n3. 音频感知记忆（模拟）:")
        
        audio_perceptions = [
            {
                "content": "一段优美的古典音乐演奏",
                "modality": "audio",
                "file_path": "/simulated/classical.mp3",
                "genre": "classical",
                "instruments": ["piano", "violin", "cello"],
                "tempo": "andante",
                "emotion": "elegant",
                "duration_seconds": 240
            },
            {
                "content": "技术会议的录音，讨论AI发展趋势",
                "modality": "audio",
                "file_path": "/simulated/conference.wav",
                "genre": "speech",
                "topic": "artificial_intelligence",
                "speakers": 3,
                "language": "chinese",
                "duration_seconds": 1800
            }
        ]
        
        for perception in audio_perceptions:
            result = self.perceptual_memory_tool.execute("add",
                                                       content=perception["content"],
                                                       memory_type="perceptual",
                                                       importance=0.7,
                                                       **{k: v for k, v in perception.items() if k != "content"})
            print(f"  音频感知: {perception['genre']} - {result}")
        
        # 演示跨模态检索
        print(f"\n4. 跨模态检索演示:")
        
        cross_modal_queries = [
            ("美丽的风景", "寻找视觉美感相关内容"),
            ("技术文档", "查找技术相关的多模态内容"),
            ("音乐和艺术", "检索艺术相关的感知记忆"),
            ("会议和讨论", "查找交流相关的内容")
        ]
        
        for query, description in cross_modal_queries:
            results = self.perceptual_memory_tool.execute("search",
                                                        query=query,
                                                        memory_type="perceptual",
                                                        limit=8)
            print(f"  跨模态查询: '{query}' ({description})")
            print(f"  结果: {results[:300]}...")
        
        # 演示感知特征分析
        print(f"\n5. 感知特征分析:")
        
        # 获取感知记忆统计
        perceptual_stats = self.perceptual_memory_tool.execute("stats")
        print(f"感知记忆统计: {perceptual_stats}")
        
        # 分析不同模态的分布
        modality_analysis = self.perceptual_memory_tool.execute("search",
                                                            #   query="模态分析",
                                                            query="古典音乐",
                                                            memory_type="perceptual",
                                                            limit=10)
        print(f"模态分布分析: {modality_analysis}")
    
    def demonstrate_memory_interactions(self):
        """演示四种记忆类型的交互"""
        print("\n🔄 四种记忆类型交互演示")
        print("-" * 60)
        
        print("🔍 记忆交互模式:")
        print("• 🔄 工作记忆 → 情景记忆（重要事件固化）")
        print("• 📚 情景记忆 → 语义记忆（经验抽象化）")
        print("• 👁️ 感知记忆 → 其他记忆（多模态信息整合）")
        print("• 🧠 语义记忆 → 工作记忆（知识激活）")
        
        # 模拟一个完整的学习过程
        print(f"\n完整学习过程模拟:")
        
        # 1. 感知阶段：接收多模态信息
        print(f"\n1. 感知阶段 - 接收信息:")
        
        perceptual_input = self.perceptual_memory_tool.execute("add",
                                                             content="观看了一个关于深度学习的视频教程",
                                                             memory_type="perceptual",
                                                             importance=0.8,
                                                             modality="video",
                                                             topic="deep_learning",
                                                             duration_minutes=45,
                                                             quality="high")
        print(f"感知记忆: {perceptual_input}")
        
        # 2. 工作记忆阶段：临时处理和思考
        print(f"\n2. 工作记忆阶段 - 临时处理:")
        
        working_thoughts = [
            "理解了卷积神经网络的基本原理",
            "需要记住反向传播的计算步骤",
            "想到了之前学过的线性代数知识",
            "计划实现一个简单的CNN模型"
        ]
        
        for thought in working_thoughts:
            result = self.working_memory_tool.execute("add",
                                                    content=thought,
                                                    memory_type="working",
                                                    importance=0.6,
                                                    processing_stage="active_thinking")
            print(f"  工作记忆: {thought[:100]}... - {result}")
        
        # 3. 情景记忆阶段：记录完整学习事件
        print(f"\n3. 情景记忆阶段 - 事件记录:")
        
        episodic_event = self.episodic_memory_tool.execute("add",
                                                         content="完成了深度学习视频教程的学习，理解了CNN的核心概念",
                                                         memory_type="episodic",
                                                         importance=0.9,
                                                         event_type="learning_session",
                                                         duration_minutes=45,
                                                         location="家里",
                                                         learning_outcome="理解CNN原理",
                                                         next_action="实践编程")
        print(f"情景记忆: {episodic_event}")
        
        # 4. 语义记忆阶段：抽象知识存储
        print(f"\n4. 语义记忆阶段 - 知识抽象:")
        
        semantic_knowledge = [
            {
                "content": "卷积神经网络通过卷积层提取图像特征，适合计算机视觉任务",
                "concept": "CNN",
                "domain": "deep_learning",
                "application": "computer_vision"
            },
            {
                "content": "反向传播算法通过链式法则计算梯度，用于更新网络参数",
                "concept": "backpropagation",
                "domain": "optimization",
                "mathematical_basis": "chain_rule"
            }
        ]
        
        for knowledge in semantic_knowledge:
            result = self.semantic_memory_tool.execute("add",
                                                     content=knowledge["content"],
                                                     memory_type="semantic",
                                                     importance=0.8,
                                                     **{k: v for k, v in knowledge.items() if k != "content"})
            print(f"  语义记忆: {knowledge['concept']} - {result}")
        
        # 5. 记忆整合演示
        print(f"\n5. 记忆整合演示:")
        
        # 从工作记忆整合到情景记忆
        consolidation_result = self.working_memory_tool.execute("consolidate",
                                                              from_type="working",
                                                              to_type="episodic",
                                                              importance_threshold=0.6)
        print(f"工作记忆整合: {consolidation_result}")
        
        # 跨记忆类型检索
        print(f"\n6. 跨记忆类型检索:")
        
        query = "深度学习CNN"
        
        # 在所有记忆类型中搜索
        memory_tools = [
            ("工作记忆", self.working_memory_tool),
            ("情景记忆", self.episodic_memory_tool),
            ("语义记忆", self.semantic_memory_tool),
            ("感知记忆", self.perceptual_memory_tool)
        ]
        
        for memory_name, tool in memory_tools:
            results = tool.execute("search", query=query, limit=2)
            print(f"  {memory_name}检索: {results[:300]}...")
        
        # 获取所有记忆系统的统计
        print(f"\n7. 系统整体状态:")
        
        for memory_name, tool in memory_tools:
            stats = tool.execute("stats")
            print(f"  {memory_name}: {stats}")

def main():
    """主函数"""
    print("🧠 四种记忆类型深度解析演示")
    print("详细展示WorkingMemory、EpisodicMemory、SemanticMemory、PerceptualMemory")
    print("=" * 80)
    
    try:
        demo = MemoryTypesDeepDive()
        
        # 1. 工作记忆演示
        demo.demonstrate_working_memory()
        
        # 2. 情景记忆演示
        demo.demonstrate_episodic_memory()
        
        # 3. 语义记忆演示
        demo.demonstrate_semantic_memory()
        
        # 4. 感知记忆演示
        demo.demonstrate_perceptual_memory()
        
        # 5. 记忆交互演示
        demo.demonstrate_memory_interactions()
        
        print("\n" + "=" * 80)
        print("🎉 四种记忆类型深度解析完成！")
        print("=" * 80)
        
        print("\n✨ 记忆类型特性总结:")
        print("1. 💭 工作记忆 - 快速临时存储，容量有限，自动过期")
        print("2. 📖 情景记忆 - 完整事件记录，时间序列，丰富上下文")
        print("3. 🧠 语义记忆 - 抽象知识存储，概念关系，语义推理")
        print("4. 👁️ 感知记忆 - 多模态支持，跨模态检索，感知理解")
        
        print("\n🔄 记忆交互模式:")
        print("• 感知 → 工作 → 情景 → 语义（信息处理流程）")
        print("• 语义 → 工作（知识激活和应用）")
        print("• 跨类型检索和整合（智能记忆管理）")
        
        print("\n💡 设计价值:")
        print("• 模拟人类认知过程")
        print("• 支持多层次信息处理")
        print("• 实现智能记忆管理")
        print("• 提供丰富的检索能力")
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
"""
TODO:
- # query "任务" 未包含 "[工作记忆] 临时工作记忆 1: 当前正在处理任务步骤 1" ? 存在默认最小importance ?
- ✅ 任务步骤 1 (重要性: 0.30) ～ 任务步骤 8 (重要性: 1.00) +  ("刚刚的想法", 0.8), ("5分钟前的任务", 0.6), ("10分钟前的提醒", 0.4), ("很久以前的笔记", 0.2)
12 working memory, forget(strategy="importance_based", threshold=0.6), 最后剩7条 ✅
- demonstrate_perceptual_memory 跨模态查询时, 为何 '技术文档：API接口返回JSON格式数据，包含状态码和响应体 (重要性: 0.70)' 
会出现在 '美丽的风景' 检索结果中; 同样 '这是一段优美的诗歌：春江潮水连海平，海上明月共潮生 (重要性: 0.70)' 
出现在 '技术文档' 检索结果中?
- episodic search 结果相关性较差: 评分机制问题? 向量搜索? query=乌兰巴托的夜 应该返回 0 条!?
2. 时间序列检索演示:
query=学习, hits=[{'id': '8a920163-04a8-44d2-bcdf-1ba09996af08', 'score': 0.60531336, 'metadata': {'memory_id': '8a920163-04a8-44d2-bcdf-1ba09996af08', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.8, 'session_id': 'session_20251215_173019', 'content': '总结今天的学习收获', 'timestamp': 1765780221, 'added_at': 1765780221}}, {'id': '3bc9a6f4-e2a4-405d-a09f-b1f96df129d2', 'score': 0.57686496, 'metadata': {'memory_id': '3bc9a6f4-e2a4-405d-a09f-b1f96df129d2', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.6, 'session_id': 'session_20251215_173019', 'content': '完成了课后练习题', 'timestamp': 1765780220, 'added_at': 1765780220}}, {'id': '512022ca-3be1-40df-bc6a-cdcea364ae09', 'score': 0.49989396, 'metadata': {'memory_id': '512022ca-3be1-40df-bc6a-cdcea364ae09', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.8, 'session_id': 'session_20251215_173019', 'content': '学习了线性回归的数学原理', 'timestamp': 1765780220, 'added_at': 1765780220}}, {'id': 'f98309a2-0396-4126-99b3-11414c30a30e', 'score': 0.45066798, 'metadata': {'memory_id': 'f98309a2-0396-4126-99b3-11414c30a30e', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.7, 'session_id': 'session_20251215_173019', 'content': '开始学习Python机器学习', 'timestamp': 1765780220, 'added_at': 1765780220}}, {'id': '80c13b7c-dc97-4538-870a-08508e3a0cb7', 'score': 0.32387137, 'metadata': {'memory_id': '80c13b7c-dc97-4538-870a-08508e3a0cb7', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.9, 'session_id': 'session_20251215_173019', 'content': '实现了第一个线性回归模型', 'timestamp': 1765780220, 'added_at': 1765780220}}]
学习时间线: 🔍 找到 5 条相关记忆:
1. [情景记忆] 实现了第一个线性回归模型 (重要性: 0.90)
2. [情景记忆] 总结今天的学习收获 (重要性: 0.80)
3. [情景记忆] 学习了线性回归的数学原理 (重要性: 0.80)
4. [情景记忆] 开始学习Python机器学习 (重要性: 0.70)
5. [情景记忆] 完成了课后练习题 (重要性: 0.60)
query=乌兰巴托的夜, hits=[{'id': '3bc9a6f4-e2a4-405d-a09f-b1f96df129d2', 'score': 0.2744857, 'metadata': {'memory_id': '3bc9a6f4-e2a4-405d-a09f-b1f96df129d2', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.6, 'session_id': 'session_20251215_173019', 'content': '完成了课后练习题', 'timestamp': 1765780220, 'added_at': 1765780220}}, {'id': '8a920163-04a8-44d2-bcdf-1ba09996af08', 'score': 0.2530401, 'metadata': {'memory_id': '8a920163-04a8-44d2-bcdf-1ba09996af08', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.8, 'session_id': 'session_20251215_173019', 'content': '总结今天的学习收获', 'timestamp': 1765780221, 'added_at': 1765780221}}, {'id': '512022ca-3be1-40df-bc6a-cdcea364ae09', 'score': 0.21372983, 'metadata': {'memory_id': '512022ca-3be1-40df-bc6a-cdcea364ae09', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.8, 'session_id': 'session_20251215_173019', 'content': '学习了线性回归的数学原理', 'timestamp': 1765780220, 'added_at': 1765780220}}, {'id': 'f98309a2-0396-4126-99b3-11414c30a30e', 'score': 0.17790505, 'metadata': {'memory_id': 'f98309a2-0396-4126-99b3-11414c30a30e', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.7, 'session_id': 'session_20251215_173019', 'content': '开始学习Python机器学习', 'timestamp': 1765780220, 'added_at': 1765780220}}, {'id': '80c13b7c-dc97-4538-870a-08508e3a0cb7', 'score': 0.15467688, 'metadata': {'memory_id': '80c13b7c-dc97-4538-870a-08508e3a0cb7', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.9, 'session_id': 'session_20251215_173019', 'content': '实现了第一个线性回归模型', 'timestamp': 1765780220, 'added_at': 1765780220}}]
会话内容: 🔍 找到 5 条相关记忆:
1. [情景记忆] 实现了第一个线性回归模型 (重要性: 0.90)
2. [情景记忆] 总结今天的学习收获 (重要性: 0.80)
3. [情景记忆] 学习了线性回归的数学原理 (重要性: 0.80)
4. [情景记忆] 开始学习Python机器学习 (重要性: 0.70)
5. [情景记忆] 完成了课后练习题 (重要性: 0.60)
- test code: 用上面的例子做向量检索, 并计算最终score
- 记忆链条检索 是如何工作的?
query=GPT Transformer, hits=[{'id': 'cc43c4c8-e570-4596-88fc-93dc4028fa58', 'score': 0.5316789, 'metadata': {'memory_id': 'cc43c4c8-e570-4596-88fc-93dc4028fa58', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.7, 'session_id': 'session_20251215_191028', 'content': '看到一篇关于GPT的论文', 'timestamp': 1765786230, 'added_at': 1765786230}}, {'id': '4fc2fa2e-9410-48fd-bf1b-7afa1862c595', 'score': 0.49412897, 'metadata': {'memory_id': '4fc2fa2e-9410-48fd-bf1b-7afa1862c595', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.7, 'session_id': 'session_20251215_191028', 'content': '决定深入研究Transformer架构', 'timestamp': 1765786231, 'added_at': 1765786231}}, {'id': 'ccdf142b-9821-49c4-96d9-c7395da6435a', 'score': 0.3488748, 'metadata': {'memory_id': 'ccdf142b-9821-49c4-96d9-c7395da6435a', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.7, 'session_id': 'session_20251215_191028', 'content': '开始学习Python机器学习', 'timestamp': 1765786228, 'added_at': 1765786228}}, {'id': 'e396d15b-0fca-468f-bba2-093dbee342aa', 'score': 0.3199976, 'metadata': {'memory_id': 'e396d15b-0fca-468f-bba2-093dbee342aa', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.9, 'session_id': 'session_20251215_191028', 'content': '参加了AI技术分享会', 'timestamp': 1765786230, 'added_at': 1765786230}}, {'id': '63c54aab-a2d8-4701-95b9-596167ddbd89', 'score': 0.2960202, 'metadata': {'memory_id': '63c54aab-a2d8-4701-95b9-596167ddbd89', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.7, 'session_id': 'session_20251215_191028', 'content': '实现了简化版的自注意力机制', 'timestamp': 1765786231, 'added_at': 1765786231}}, {'id': 'bef54ec6-0ea7-47c2-b2fb-c2b2d5554b64', 'score': 0.27358294, 'metadata': {'memory_id': 'bef54ec6-0ea7-47c2-b2fb-c2b2d5554b64', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.7, 'session_id': 'session_20251215_191028', 'content': '下载并阅读Attention is All You Need论文', 'timestamp': 1765786231, 'added_at': 1765786231}}, {'id': '1a5ce807-7de0-455b-9ebb-ec4efb69ca45', 'score': 0.25967684, 'metadata': {'memory_id': '1a5ce807-7de0-455b-9ebb-ec4efb69ca45', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.7, 'session_id': 'session_20251215_191028', 'content': '在项目中应用了学到的知识', 'timestamp': 1765786232, 'added_at': 1765786232}}, {'id': '4d1869e1-5f42-4009-9d4e-6fb7dc20a586', 'score': 0.2550019, 'metadata': {'memory_id': '4d1869e1-5f42-4009-9d4e-6fb7dc20a586', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.8, 'session_id': 'session_20251215_191028', 'content': '学习了线性回归的数学原理', 'timestamp': 1765786228, 'added_at': 1765786228}}, {'id': '15ba8481-8dfe-4db2-8058-11ad49295a3b', 'score': 0.24384682, 'metadata': {'memory_id': '15ba8481-8dfe-4db2-8058-11ad49295a3b', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.9, 'session_id': 'session_20251215_191028', 'content': '实现了第一个线性回归模型', 'timestamp': 1765786228, 'added_at': 1765786228}}, {'id': '0d1f2ca9-a5c5-4398-a33f-f7296bcf2c9f', 'score': 0.21660641, 'metadata': {'memory_id': '0d1f2ca9-a5c5-4398-a33f-f7296bcf2c9f', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.8, 'session_id': 'session_20251215_191028', 'content': '总结今天的学习收获', 'timestamp': 1765786229, 'added_at': 1765786229}}, {'id': 'bca81a07-4184-4824-926a-d378de9b33b9', 'score': 0.21339527, 'metadata': {'memory_id': 'bca81a07-4184-4824-926a-d378de9b33b9', 'user_id': 'episodic_memory_user', 'memory_type': 'episodic', 'importance': 0.6, 'session_id': 'session_20251215_191028', 'content': '完成了课后练习题', 'timestamp': 1765786229, 'added_at': 1765786229}}]
记忆链条检索: 🔍 找到 11 条相关记忆:

- neo4j (NER + RE) verify: 为什么 semantic:✅ 添加语义记忆: 0个实体, 0个关系 ? bug?
DEBUG:hello_agents.memory.types.semantic:🌐 检测语言: zh, 使用模型: core_web_sm
DEBUG:hello_agents.memory.types.semantic:📝 spaCy处理文本: '机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式' -> 0 个实体
DEBUG:hello_agents.memory.storage.neo4j_store:✅ 添加实体: 机器 (TOKEN)
DEBUG:hello_agents.memory.storage.neo4j_store:✅ 添加实体: 机器 (CONCEPT)
DEBUG:hello_agents.memory.storage.neo4j_store:✅ 添加关系: token_-8494211641031686787 -REPRESENTS-> concept_-7766050944591470752
... ...
DEBUG:hello_agents.memory.types.semantic:🔗 已将词法分析结果存储到Neo4j: 17 个词元
DEBUG:hello_agents.memory.types.semantic:🔍 未找到实体，词元分析:
DEBUG:hello_agents.memory.types.semantic:   '机器' -> POS: NOUN, TAG: NN, ENT_IOB: O
DEBUG:hello_agents.memory.types.semantic:   '学习' -> POS: NOUN, TAG: NN, ENT_IOB: O
.. ...
INFO:hello_agents.memory.types.semantic:✅ 添加语义记忆: 0个实体, 0个关系
DEBUG:hello_agents.memory.manager:添加记忆到 semantic: 72a22ec3-886c-4f11-b988-59f20d8521a0
机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式: [], []
MEMORY_ITEM: id='72a22ec3-886c-4f11-b988-59f20d8521a0' content='机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式' memory_type='semantic' user_id='semantic_memory_user' timestamp=datetime.datetime(2025, 12, 15, 19, 10, 32, 560120) importance=0.9 metadata={'concept_type': 'definition', 'domain': 'artificial_intelligence', 'keywords': ['机器学习', '人工智能', '算法', '数据', '模式'], 'session_id': 'session_20251215_191032', 'timestamp': '2025-12-15T19:10:32.560073', 'entities': [], 'relations': []}
  概念存储: definition - ✅ 记忆已添加 (ID: 72a22ec3...)

- demonstrate_semantic_memory 中 语义相似度检索:
vector_results=[{'id': '9d6c786b-04dc-459e-a6bb-43dcc4d6abae', 'score': 0.53965545, 'memory_id': '9d6c786b-04dc-459e-a6bb-43dcc4d6abae', 'user_id': 'semantic_memory_user', 'content': '机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式', 'memory_type': 'semantic', 'timestamp': 1765788545, 'importance': 0.9, 'entities': [], 'entity_count': 0, 'relation_count': 0, 'added_at': 1765788545}, {'id': '6c663eb4-6ac2-4531-aa84-d17a491f4e5d', 'score': 0.43146944, 'memory_id': '6c663eb4-6ac2-4531-aa84-d17a491f4e5d', 'user_id': 'semantic_memory_user', 'content': '深度学习是机器学习的子集，使用多层神经网络', 'memory_type': 'semantic', 'timestamp': 1765788548, 'importance': 0.8, 'entities': [], 'entity_count': 0, 'relation_count': 0, 'added_at': 1765788548}, {'id': 'a6b0bd09-24ba-49e6-a91d-d1912961bf88', 'score': 0.3003248, 'memory_id': 'a6b0bd09-24ba-49e6-a91d-d1912961bf88', 'user_id': 'semantic_memory_user', 'content': '过拟合是指模型在训练数据上表现很好，但在新数据上泛化能力差', 'memory_type': 'semantic', 'timestamp': 1765788548, 'importance': 0.7, 'entities': [], 'entity_count': 0, 'relation_count': 0, 'added_at': 1765788548}, {'id': '968e9f15-15e9-4c1b-b6c4-5ac9ab151f13', 'score': 0.2995221, 'memory_id': '968e9f15-15e9-4c1b-b6c4-5ac9ab151f13', 'user_id': 'semantic_memory_user', 'content': '梯度下降是一种优化算法，通过迭代更新参数来最小化损失函数', 'memory_type': 'semantic', 'timestamp': 1765788547, 'importance': 0.8, 'entities': [], 'entity_count': 0, 'relation_count': 0, 'added_at': 1765788547}, {'id': 'c0cc6970-e088-4067-ba14-49fb4fa480c9', 'score': 0.2898135, 'memory_id': 'c0cc6970-e088-4067-ba14-49fb4fa480c9', 'user_id': 'semantic_memory_user', 'content': '卷积神经网络特别适合处理图像数据', 'memory_type': 'semantic', 'timestamp': 1765788549, 'importance': 0.8, 'entities': [], 'entity_count': 0, 'relation_count': 0, 'added_at': 1765788549}, {'id': '251ea491-c951-46ec-8ece-36ec6f9c210b', 'score': 0.25019962, 'memory_id': '251ea491-c951-46ec-8ece-36ec6f9c210b', 'user_id': 'semantic_memory_user', 'content': '反向传播算法用于训练神经网络', 'memory_type': 'semantic', 'timestamp': 1765788550, 'importance': 0.8, 'entities': [], 'entity_count': 0, 'relation_count': 0, 'added_at': 1765788550}, {'id': 'b1f1318f-7fd2-44c1-9c00-ad418c0a43f6', 'score': 0.23151018, 'memory_id': 'b1f1318f-7fd2-44c1-9c00-ad418c0a43f6', 'user_id': 'semantic_memory_user', 'content': '监督学习使用标记数据训练模型，包括分类和回归两大类任务', 'memory_type': 'semantic', 'timestamp': 1765788546, 'importance': 0.8, 'entities': ['entity_-9024021128637848739'], 'entity_count': 1, 'relation_count': 0, 'added_at': 1765788546}]
graph_results=[]
neo4j搜索没有结果返回应该和之前 NER + RE 未找到实体有关?
对 查询 '太极拳分哪些流派?' vector_results 应该返回 [] ?

- demonstrate_perceptual_memory 中 跨模态检索 结果很不准确 ?!

- demonstrate_memory_interactions 中 语义记忆阶段 - 知识抽象 抛出警告:
WARNING:hello_agents.memory.manager:记忆类型不存在: working -> episodic
"""