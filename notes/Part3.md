# 第三部分：高级知识扩展
## [第八章 记忆与检索](https://datawhalechina.github.io/hello-agents/#/./chapter8/第八章%20记忆与检索)
为HelloAgents增加两个核心能力：**记忆系统（Memory System）** 和 **检索增强生成（Retrieval-Augmented Generation, RAG）**。
### 8.1 从认知科学到智能体记忆
#### 8.1.1 人类记忆系统的启发

![人类记忆系统的层次结构](../docs/images/8-figures/8-1.png)
<center>图 8.1 人类记忆系统的层次结构</center>

根据认知心理学的研究，人类记忆可以分为以下几个层次：

1. **感觉记忆（Sensory Memory）**：持续时间极短（0.5-3秒），容量巨大，负责暂时保存感官接收到的所有信息
2. **工作记忆（Working Memory）**：持续时间短（15-30秒），容量有限（7±2个项目），负责当前任务的信息处理
3. **长期记忆（Long-term Memory）**：持续时间长（可达终生），容量几乎无限，进一步分为：
  - **程序性记忆**：技能和习惯（如骑自行车）
  - **陈述性记忆**：可以用语言表达的知识，又分为：
    * **语义记忆**：一般知识和概念（如"巴黎是法国首都"）
    * **情景记忆**：个人经历和事件（如"昨天的会议内容"）

#### 8.1.2 为何智能体需要记忆与RAG
对于基于LLM的智能体而言，通常面临两个根本性局限：**对话状态的遗忘**和**内置知识的局限**。
(1) 局限一：无状态导致的对话遗忘
当前的大语言模型虽然强大，但设计上是**无状态**的。
- **上下文丢失**：在长对话中，早期的重要信息可能会因为上下文窗口限制而丢失
- **个性化缺失**：Agent无法记住用户的偏好、习惯或特定需求
- **学习能力受限**：无法从过往的成功或失败经验中学习改进
- **一致性问题**：在多轮对话中可能出现前后矛盾的回答

(2) 局限二：模型内置知识的局限性
除了遗忘对话历史，LLM 的另一个核心局限在于其知识是静态的、有限的。
- **知识时效性**：大模型的训练数据有时间截止点，无法获取最新信息
- **专业领域知识**：通用模型在特定领域的深度知识可能不足
- **事实准确性**：通过检索验证，减少模型的幻觉问题
- **可解释性**：提供信息来源，增强回答的可信度

#### 8.1.3 记忆与RAG系统架构设计
![HelloAgents记忆与RAG系统整体架构](../docs/images/8-figures/8-2.png)
<center>图 8.2 HelloAgents记忆与RAG系统整体架构</center>

记忆系统采用了四层架构设计：
```
HelloAgents记忆系统
├── 基础设施层 (Infrastructure Layer)
│   ├── MemoryManager - 记忆管理器（统一调度和协调）
│   ├── MemoryItem - 记忆数据结构（标准化记忆项）
│   ├── MemoryConfig - 配置管理（系统参数设置）
│   └── BaseMemory - 记忆基类（通用接口定义）
├── 记忆类型层 (Memory Types Layer)
│   ├── WorkingMemory - 工作记忆（临时信息，TTL管理）
│   ├── EpisodicMemory - 情景记忆（具体事件，时间序列）
│   ├── SemanticMemory - 语义记忆（抽象知识，图谱关系）
│   └── PerceptualMemory - 感知记忆（多模态数据）
├── 存储后端层 (Storage Backend Layer)
│   ├── QdrantVectorStore - 向量存储（高性能语义检索）
│   ├── Neo4jGraphStore - 图存储（知识图谱管理）
│   └── SQLiteDocumentStore - 文档存储（结构化持久化）
└── 嵌入服务层 (Embedding Service Layer)
    ├── DashScopeEmbedding - 通义千问嵌入（云端API）
    ├── LocalTransformerEmbedding - 本地嵌入（离线部署）
    └── TFIDFEmbedding - TFIDF嵌入（轻量级兜底）
```
RAG系统专注于外部知识的获取和利用：
```
HelloAgents RAG系统
├── 文档处理层 (Document Processing Layer)
│   ├── DocumentProcessor - 文档处理器（多格式解析）
│   ├── Document - 文档对象（元数据管理）
│   └── Pipeline - RAG管道（端到端处理）
├── 嵌入表示层 (Embedding Layer)
│   └── 统一嵌入接口 - 复用记忆系统的嵌入服务
├── 向量存储层 (Vector Storage Layer)
│   └── QdrantVectorStore - 向量数据库（命名空间隔离）
└── 智能问答层 (Intelligent Q&A Layer)
    ├── 多策略检索 - 向量检索 + MQE + HyDE
    ├── 上下文构建 - 智能片段合并与截断
    └── LLM增强生成 - 基于上下文的准确问答
```

#### 8.1.4 本章学习目标与快速体验
```bash
pip install neo4j
# search is removed since 1.16.0
micromamba install qdrant-client==1.15.1 spacy
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

### 8.2 记忆系统：让智能体拥有记忆
#### 8.2.1 记忆系统的工作流程
![记忆形成的认知过程](../docs/images/8-figures/8-3.png)
<center>图 8.3 记忆形成的认知过程</center>

根据认知科学的研究，人类记忆的形成经历以下几个阶段：
- **编码（Encoding）**：将感知到的信息转换为可存储的形式
- **存储（Storage）**：将编码后的信息保存在记忆系统中
- **检索（Retrieval）**：根据需要从记忆中提取相关信息
- **整合（Consolidation）**：将短期记忆转化为长期记忆
- **遗忘（Forgetting）**：删除不重要或过时的信息

基于该启发，我们为 HelloAgents 设计了一套完整的记忆系统。其核心思想是模仿人类大脑处理不同类型信息的方式，将记忆划分为多个专门的模块，并建立一套智能化的管理机制。图8.4详细展示了这套系统的工作流程，包括记忆的添加、检索、整合和遗忘等关键环节。

![HelloAgents记忆系统的完整工作流程](../docs/images/8-figures/8-4.png)
<center>图 8.4 HelloAgents记忆系统的完整工作流程</center>

-- **工作记忆 (Working Memory)**: 智能体“短期记忆”，主要用于存储当前对话的上下文信息。为确保高速访问和响应，其容量被有意限制（例如，默认50条），并且生命周期与单个会话绑定，会话结束后便会自动清理。
-- **情景记忆 (Episodic Memory)**: 负责长期存储具体的交互事件和智能体的学习经历。与工作记忆不同，情景记忆包含了丰富的上下文信息，并支持按时间序列或主题进行回顾式检索，是智能体“复盘”和学习过往经验的基础。
-- **语义记忆 (Semantic Memory)**: 存储的是更为抽象的知识、概念和规则。例如，通过对话了解到的用户偏好、需要长期遵守的指令或领域知识点，都适合存放在这里。这部分记忆具有高度的持久性和重要性，是智能体形成“知识体系”和进行关联推理的核心。
-- **感知记忆 (Perceptual Memory)**: 该模块专门处理图像、音频等多模态信息，并支持跨模态检索。其生命周期会根据信息的重要性和可用存储空间进行动态管理。

#### 8.2.2 快速体验：30秒上手记忆功能
#### 8.2.3 MemoryTool详解
- (1) 操作1：add
- (2) 操作2：search
- (3) 操作3：forget, 三种遗忘策略: 基于重要性(importance_based); 基于时间(time_based); 基于容量(capacity_based)
- (4) 操作4：consolidate, 模拟人类大脑将短期记忆转化为长期记忆的过程

#### 8.2.4 MemoryManager详解
#### 8.2.5 四种记忆类型
- (1) 工作记忆（WorkingMemory）
  * 容量有限（默认50条）+ TTL自动清理; 纯内存存储，访问速度极快; 混合检索：TF-IDF向量化 + 关键词匹配
  * 评分算法结合了语义相似度、时间衰减和重要性权重，最终得分公式为：`(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)`。
- (2) 情景记忆（EpisodicMemory）
  * 负责存储具体的事件和经历，它的设计重点在于保持事件的完整性和时间序列关系。SQLite+Qdrant
  * 评分公式为：`(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)`，确保检索结果既语义相关又时间相关。
- (3) 语义记忆（SemanticMemory）: 
  * 语义记忆是记忆系统中最复杂的部分，它负责存储抽象的概念、规则和知识。语义记忆采用了Neo4j图数据库和Qdrant向量数据库的混合架构
  * 语义记忆的评分公式为：`(向量相似度 × 0.7 + 图相似度 × 0.3) × (0.8 + 重要性 × 0.4)`。
  * **向量检索权重（0.7）**：语义相似度是主要因素，确保检索结果与查询语义相关
  * **图检索权重（0.3）**：关系推理作为补充，发现概念间的隐含关联
  * **重要性权重范围[0.8, 1.2]**：避免重要性过度影响相似度排序，保持检索的准确性
- (4) 感知记忆（PerceptualMemory）
  * 感知记忆支持文本、图像、音频等多种模态的数据存储和检索。
  * 感知记忆的评分公式为：`(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4) `
  * 感知记忆中的时间近因性计算采用了指数衰减模型，模拟了人类记忆中的遗忘曲线

### 8.3 RAG系统：知识检索增强
#### 8.3.1 RAG的基础知识
(1) 什么是RAG？
检索增强生成（Retrieval-Augmented Generation，RAG）是一种结合了信息检索和文本生成的技术。它的核心思想是：在生成回答之前，先从外部知识库中检索相关信息，然后将检索到的信息作为上下文提供给大语言模型，从而生成更准确、更可靠的回答。

(2) 基本工作流程
(3) 发展历程
- 第一阶段：朴素RAG（Naive RAG, 2020-2021）。检索方式：主要依赖传统的关键词匹配算法，如TF-IDF或BM25。生成模式：将检索到的文档内容不加处理地直接拼接到提示词的上下文中，然后送给生成模型。
- 第二阶段：高级RAG（Advanced RAG, 2022-2023）。检索方式：转向基于稠密嵌入（Dense Embedding）的语义检索。生成模式：引入了很多优化技术，例如查询重写，文档分块，重排序等。
- 第三阶段：模块化RAG（Modular RAG, 2023-至今）。检索方式：如混合检索，多查询扩展，假设性文档嵌入等。生成模式：思维链推理，自我反思与修正等。

#### 8.3.2 RAG系统工作原理
![RAG系统的核心工作原理](../docs/images/8-figures/8-5.png)
<center>图 8.5 RAG系统的核心工作原理</center>

如图8.5所示，展示了RAG系统的两个主要工作模式：
- **数据处理流程**：处理和存储知识文档，在这里我们采取工具Markitdown，设计思路是将传入的一切外部知识源统一转化为Markdown格式进行处理。
- **查询与生成流程**：根据查询检索相关信息并生成回答。

#### 8.3.3 快速体验：30秒上手RAG功能
[t05.py::t03](t05.py)

#### 8.3.4 RAG系统架构设计
[RAGTool](https://github.com/jjyaoao/HelloAgents/blob/main/hello_agents/tools/builtin/rag_tool.py)
我们的RAG系统的核心架构可以概括为"五层七步"的设计模式：
```
用户层：RAGTool统一接口
  ↓
应用层：智能问答、搜索、管理
  ↓  
处理层：文档解析、分块、向量化
  ↓
存储层：向量数据库、文档存储
  ↓
基础层：嵌入模型、LLM、数据库
```
整个处理流程：任意格式文档 → MarkItDown转换 → Markdown文本 → 智能分块 → 向量化 → 存储检索

(1) 多模态文档载入
- [_convert_to_markdown](https://github.com/jjyaoao/HelloAgents/blob/main/hello_agents/memory/rag/pipeline.py#L49)
- [t7.py](t07.py): png => md (OCR 不工作); pdf => md (列表、表格等格式丢失，内容还行)

(2) 智能分块策略
Markdown结构感知的分块流程：
```
标准Markdown文本 → 标题层次解析 → 段落语义分割 → Token计算分块 → 重叠策略优化 → 向量化准备
       ↓                ↓              ↓            ↓           ↓            ↓
   统一格式          #/##/###        语义边界      大小控制     信息连续性    嵌入向量
   结构清晰          层次识别        完整性保证    检索优化     上下文保持    相似度匹配
```
- [_split_paragraphs_with_headings](https://github.com/jjyaoao/HelloAgents/blob/main/hello_agents/memory/rag/pipeline.py#L227)
- [_chunk_paragraphs](https://github.com/jjyaoao/HelloAgents/blob/main/hello_agents/memory/rag/pipeline.py#L272)
- [_approx_token_len](https://github.com/jjyaoao/HelloAgents/blob/main/hello_agents/memory/rag/pipeline.py#L220)

(3) 统一嵌入与向量存储
- [index_chunks](https://github.com/jjyaoao/HelloAgents/blob/main/hello_agents/memory/rag/pipeline.py#L480)

#### 8.3.5 高级检索策略
RAG系统的检索能力是其核心竞争力。在实际应用中，用户的查询表述与文档中的实际内容可能存在用词差异，导致相关文档无法被检索到。为了解决这个问题，HelloAgents实现了三种互补的高级检索策略：多查询扩展（MQE）、假设文档嵌入（HyDE）和统一的扩展检索框架。
(1) 多查询扩展（MQE）
多查询扩展（Multi-Query Expansion）是一种通过生成语义等价的多样化查询来提高检索召回率的技术。
- [_prompt_mqe](https://github.com/jjyaoao/HelloAgents/blob/main/hello_agents/memory/rag/pipeline.py#L712)

(2) 假设文档嵌入（HyDE）
假设文档嵌入（Hypothetical Document Embeddings，HyDE）是一种创新的检索技术，它的核心思想是"用答案找答案"。
- [_prompt_hyde](https://github.com/jjyaoao/HelloAgents/blob/main/hello_agents/memory/rag/pipeline.py#L728)

(3) 扩展检索框架
扩展检索的核心机制是"扩展-检索-合并"三步流程。
- [search_vectors_expanded](https://github.com/jjyaoao/HelloAgents/blob/main/hello_agents/memory/rag/pipeline.py#L741)

实际应用中，这三种策略的组合使用效果最佳。
MQE擅长处理用词多样性问题，HyDE擅长处理语义鸿沟问题，而统一框架则确保了结果的质量和多样性。
对于一般查询，建议启用MQE；对于专业领域查询，建议同时启用MQE和HyDE；对于性能敏感场景，可以只使用基础检索或仅启用MQE。

### 8.4 构建智能文档问答助手
#### 8.4.1 案例背景与目标
- **智能文档处理**：使用MarkItDown实现PDF到Markdown的统一转换，基于Markdown结构的智能分块策略，高效的向量化和索引构建
- **高级检索问答**：多查询扩展（MQE）提升召回率，假设文档嵌入（HyDE）改善检索精度，上下文感知的智能问答
- **多层次记忆管理**：工作记忆管理当前学习任务和上下文，情景记忆记录学习事件和查询历史，语义记忆存储概念知识和理解，感知记忆处理文档特征和多模态信息
- **个性化学习支持**：基于学习历史的个性化推荐，记忆整合和选择性遗忘，学习报告生成和进度追踪

![智能问答助手的五步执行流程](../docs/images/8-figures/8-6.png)
<center>图 8.6 智能问答助手的五步执行流程</center>

整个应用分为三个核心部分：
- **核心助手类（PDFLearningAssistant）**：封装RAGTool和MemoryTool的调用逻辑
- **Gradio Web界面**：提供友好的用户交互界面，这个部分可以参考示例代码学习
- **其他核心功能**：笔记记录、学习回顾、统计查看和报告生成

TODO: 关于chapter8 中例子的一些疑问, 需要查看hello-agents 源码, 加日志调试
- 01_MemoryTool_Basic_Operations: search_memory_demo 设置重要性阈值, limit=3 改为 limit=2 可以看到结果变化, 但是改变 min_importance=0.7 貌似没起作用(重要性0.6的感知记忆也被选中?), 是否每个类别至少返回一条? 
```
高重要性记忆搜索:
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.29it/s]
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.21it/s]
INFO:hello_agents.memory.types.semantic:✅ 检索到 1 条相关记忆
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.59it/s]
🔍 找到 3 条相关记忆:
1. [语义记忆] 记忆系统包括工作记忆、情景记忆、语义记忆和感知记忆四种类型 (重要性: 0.90)
2. [情景记忆] 2024年开始深入研究AI Agent技术 (重要性: 0.80)
3. [感知记忆] 查看了记忆系统的架构图和实现代码 (重要性: 0.60)
```
- 03_WorkingMemory_Implementation.py 中混合检索测试的结果是因为 working memory 是基于TF-IDF 向量相似比较?
- 03_WorkingMemory_Implementation.py 中时间衰减效果测试的结果为何不是最新的2～4条? 结果没有包含 newest (最新的重要信息 - 刚刚学习的概念) 那条记忆
```
🔍 时间衰减效果测试:
搜索结果（注意时间因素对排序的影响）:
🔍 找到 2 条相关记忆:
1. [工作记忆] 较旧的信息 - 上周学习的内容 (重要性: 0.70)
2. [工作记忆] 较新的信息 - 昨天学习的内容 (重要性: 0.70)
```
- 03_WorkingMemory_Implementation.py 中添加低重要性记忆 为何stats 检查记录数没有变化?
- 03_WorkingMemory_Implementation.py 中执行基于重要性的清理 10条记忆的importance 应该分别为0.3 0.37 0.44 0.51 0.58 0.65 0.72 0.79 0.86 0.93, 阀值threshold=0.8时，应该清理(forget)8条才对? 还是需要根据公式`(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)`重新计算importance?
- 

        
#### 8.4.2 核心助手类的实现

### 8.5 本章总结与展望