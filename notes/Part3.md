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
- [02_MemoryTool_Architecture](../code/chapter8/02_MemoryTool_Architecture.py): MemoryTool架构设计, 展示MemoryTool和MemoryManager的分层架构

#### 8.1.4 本章学习目标与快速体验
```bash
pip install neo4j
# search is removed since 1.16.0
micromamba install qdrant-client==1.15.1 spacy
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```
第八章的核心学习内容：
```
hello-agents/
├── hello_agents/
│   ├── memory/                   # 记忆系统模块
│   │   ├── base.py               # 基础数据结构（MemoryItem, MemoryConfig, BaseMemory）
│   │   ├── manager.py            # 记忆管理器（统一协调调度）
│   │   ├── embedding.py          # 统一嵌入服务（DashScope/Local/TFIDF）
│   │   ├── types/                # 记忆类型实现
│   │   │   ├── working.py        # 工作记忆（TTL管理，纯内存）
│   │   │   ├── episodic.py       # 情景记忆（事件序列，SQLite+Qdrant）
│   │   │   ├── semantic.py       # 语义记忆（知识图谱，Qdrant+Neo4j）
│   │   │   └── perceptual.py     # 感知记忆（多模态，SQLite+Qdrant）
│   │   ├── storage/              # 存储后端实现
│   │   │   ├── qdrant_store.py   # Qdrant向量存储（高性能向量检索）
│   │   │   ├── neo4j_store.py    # Neo4j图存储（知识图谱管理）
│   │   │   └── document_store.py # SQLite文档存储（结构化持久化）
│   │   └── rag/                  # RAG系统
│   │       ├── pipeline.py       # RAG管道（端到端处理）
│   │       └── document.py       # 文档处理器（多格式解析）
│   └── tools/builtin/            # 扩展内置工具
│       ├── memory_tool.py        # 记忆工具（Agent记忆能力）
│       └── rag_tool.py           # RAG工具（智能问答能力）
└──
```
- [t05.py](t05.py): method t01

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

- **工作记忆 (Working Memory)**: 智能体“短期记忆”，主要用于存储当前对话的上下文信息。为确保高速访问和响应，其容量被有意限制（例如，默认50条），并且生命周期与单个会话绑定，会话结束后便会自动清理。
- **情景记忆 (Episodic Memory)**: 负责长期存储具体的交互事件和智能体的学习经历。与工作记忆不同，情景记忆包含了丰富的上下文信息，并支持按时间序列或主题进行回顾式检索，是智能体“复盘”和学习过往经验的基础。
- **语义记忆 (Semantic Memory)**: 存储的是更为抽象的知识、概念和规则。例如，通过对话了解到的用户偏好、需要长期遵守的指令或领域知识点，都适合存放在这里。这部分记忆具有高度的持久性和重要性，是智能体形成“知识体系”和进行关联推理的核心。
- **感知记忆 (Perceptual Memory)**: 该模块专门处理图像、音频等多模态信息，并支持跨模态检索。其生命周期会根据信息的重要性和可用存储空间进行动态管理。

#### 8.2.2 快速体验：30秒上手记忆功能
#### 8.2.3 MemoryTool详解
- (1) 操作1：add
- (2) 操作2：search
- (3) 操作3：forget, 三种遗忘策略: 基于重要性(importance_based); 基于时间(time_based); 基于容量(capacity_based)
- (4) 操作4：consolidate, 模拟人类大脑将短期记忆转化为长期记忆的过程

#### 8.2.4 MemoryManager详解
#### 8.2.5 四种记忆类型
- (1) **工作记忆（WorkingMemory）: 纯内存 + TTL**
  * 容量有限（默认50条）+ TTL自动清理; 纯内存存储，访问速度极快; 混合检索：TF-IDF向量化 + 关键词匹配
  * 评分算法结合了语义相似度、时间衰减和重要性权重，最终得分公式为：`(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)`。
- (2) **情景记忆（EpisodicMemory）: SQLite + Qdrant**
  * 负责存储具体的事件和经历，它的设计重点在于保持事件的完整性和时间序列关系。
  * 评分公式为：`(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4)`，确保检索结果既语义相关又时间相关。
- (3) **语义记忆（SemanticMemory）: Neo4j + Qdrant**
  * 语义记忆是记忆系统中最复杂的部分，它负责存储抽象的概念、规则和知识。
  * 语义记忆的评分公式为：`(向量相似度 × 0.7 + 图相似度 × 0.3) × (0.8 + 重要性 × 0.4)`。
  * **向量检索权重（0.7）**：语义相似度是主要因素，确保检索结果与查询语义相关
  * **图检索权重（0.3）**：关系推理作为补充，发现概念间的隐含关联
  * **重要性权重范围[0.8, 1.2]**：避免重要性过度影响相似度排序，保持检索的准确性
- (4) **感知记忆（PerceptualMemory）: SQLite + Qdrant**
  * 感知记忆支持文本、图像、音频等多种模态的数据存储和检索。
  * 感知记忆的评分公式为：`(向量相似度 × 0.8 + 时间近因性 × 0.2) × (0.8 + 重要性 × 0.4) `
  * 感知记忆中的时间近因性计算采用了指数衰减模型，模拟了人类记忆中的遗忘曲线

- [01_MemoryTool_Basic_Operations.py](../code/chapter8/01_MemoryTool_Basic_Operations.py): MemoryTool基础操作, 展示MemoryTool的核心execute方法和基本操作
- [03_WorkingMemory_Implementation.py](../code/chapter8/03_WorkingMemory_Implementation.py): WorkingMemory实现详解, 展示工作记忆的混合检索策略和TTL机制
- [06_Memory_Consolidation_Demo.py](../code/chapter8/06_Memory_Consolidation_Demo.py): 记忆整合机制演示, 展示从短期记忆到长期记忆的智能转化过程
- [09_Memory_Types_Deep_Dive.py](../code/chapter8/09_Memory_Types_Deep_Dive.py): 四种记忆类型深度解析, 详细展示WorkingMemory、EpisodicMemory、SemanticMemory、PerceptualMemory的实现特点


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
   结构清晰          层次识别        完整性保证      检索优化     上下文保持   相似度匹配
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

- [04_RAGTool_MarkItDown_Pipeline.py](../code/chapter8/04_RAGTool_MarkItDown_Pipeline.py): RAGTool的MarkItDown处理管道, 展示Any格式→Markdown→分块→向量化的完整流程
- [05_RAGTool_Advanced_Search.py](../code/chapter8/05_RAGTool_Advanced_Search.py): RAGTool高级检索策略, 展示MQE、HyDE等先进检索技术的实现和应用

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

#### 8.4.2 核心助手类的实现
#### 8.4.3 智能问答功能
当我们调用self.rag_tool.execute("ask", ...)时，RAGTool内部执行了以下高级检索流程：
- **多查询扩展（MQE）**
- **假设文档嵌入（HyDE）**

#### 8.4.4 其他核心功能
- **add_note**：将学习笔记保存到语义记忆
- **recall**：从记忆系统中检索学习历程
- **get_stats**：获取当前会话的统计信息
- **generate_report**：生成详细的学习报告并保存为JSON文件

#### 8.4.5 运行效果展示

### 8.5 本章总结与展望

- [07_RAGTool_Intelligent_QA.py](../code/chapter8/07_RAGTool_Intelligent_QA.py): RAGTool智能问答系统, 展示完整的检索→上下文构建→答案生成流程
- [10_RAG_Pipeline_Complete.py](../code/chapter8/10_RAG_Pipeline_Complete.py): RAG完整处理管道, 展示从文档处理到智能问答的完整RAG流程

#### TODO: 例子程序运行问题与排错
chapter8 中的例子在运行过程中碰到一些问题, 需要调试 hello-agents 源码, 确认实现细节或者通过尝试调整参数来解决 ✅️⚠️
- [01_MemoryTool_Basic_Operations.py](../code/chapter8/01_MemoryTool_Basic_Operations.py)
- [02_MemoryTool_Architecture.py](../code/chapter8/02_MemoryTool_Architecture.py)
- [03_WorkingMemory_Implementation.py](../code/chapter8/03_WorkingMemory_Implementation.py) 
- [04_RAGTool_MarkItDown_Pipeline.py](../code/chapter8/04_RAGTool_MarkItDown_Pipeline.py)
- [05_RAGTool_Advanced_Search.py](../code/chapter8/05_RAGTool_Advanced_Search.py) 
- [06_Memory_Consolidation_Demo.py](../code/chapter8/06_Memory_Consolidation_Demo.py)
- [07_RAGTool_Intelligent_QA.py](../code/chapter8/07_RAGTool_Intelligent_QA.py)
- [08_Agent_Tool_Integration.py](../code/chapter8/08_Agent_Tool_Integration.py)
- [09_Memory_Types_Deep_Dive.py](../code/chapter8/09_Memory_Types_Deep_Dive.py)
- [10_RAG_Pipeline_Complete.py](../code/chapter8/10_RAG_Pipeline_Complete.py)
- [11_Q&A_Assistant.py](../code/chapter8/11_Q&A_Assistant.py)

- [HelloAgents](https://github.com/jjyaoao/HelloAgents/) 建议:
  - episodic.py:retrieve, episode不要用.context , 可能会和外部定义的 obj 属性(09_Memory_Types_Deep_Dive.py:demonstrate_working_memory)冲突. 内部定义的属性可以考虑使用 _context_, 或者 __context__ ?
  ```python
  if episode and isinstance(episode.context, dict) and episode.context.get("forgotten", False):
    continue  # 跳过已遗忘的记忆
  ```
  - NER + RE 采用类似 ToolRegister 的方式, 允许用户接入除了 spaCy 之外的其他库, 如 HanLP, Stanza
  - Embedding 采用类似 ToolRegister 的方式, 允许用户接入其他实现, 如 Genimi models/gemini-embedding-001
  - Perceptual Memory 如果能够存入所有 metadata 信息是否更好, 方便溯源? (09_Memory_Types_Deep_Dive.py:demonstrate_perceptual_memory), 比如file_path, colors, objects, mood 等属性都未保存到 qdrant 向量数据库中, 这样无法更好提供引用溯源和语义检索?

## [第九章 上下文工程](https://datawhalechina.github.io/hello-agents/#/./chapter9/第九章%20上下文工程)
HelloAgents框架中上下文构建器和两个配套工具：
- **ContextBuilder (hello_agents/context/builder.py)**：上下文构建器，实现 GSSC (Gather-Select-Structure-Compress) 流水线，提供统一的上下文管理接口
- **NoteTool (hello_agents/tools/builtin/note_tool.py)**：结构化笔记工具，支持智能体进行持久化记忆管理
- **TerminalTool (hello_agents/tools/builtin/terminal_tool.py)**：终端工具，支持智能体进行文件系统操作和即时上下文检索

### 9.1 什么是上下文工程
![Prompt engineering vs Context engineering](../docs/images/9-figures/9-1.webp)
<center>图 9.1 Prompt engineering vs Context engineering</center>

一个循环运行的智能体，会不断产生下一轮推理可能相关的数据，这些信息必须被**周期性地提炼**。
因此，上下文工程的“艺与术”，在于从持续扩张的“候选信息宇宙”中，**甄别哪些内容应当进入有限的上下文窗口**。

### 9.2 为什么上下文工程重要
针堆找针（needle-in-a-haystack）类基准揭示了一个现象：**上下文腐蚀（context rot）**
**有意识的上下文工程**就成为构建强健智能体的必需品。

#### 9.2.1 有效上下文的“解剖学”
优秀的上下文工程目标是：**用尽可能少、但高信号密度的 tokens，最大化获得期望结果的概率**。
建议围绕以下组件开展工程化建设：
- **系统提示（System Prompt）**：语言清晰、直白，信息层级把握在“刚刚好”的高度。
- **工具（Tools）**：工具定义了智能体与信息/行动空间的契约，必须促进效率：既要返回token 友好的信息，又要鼓励高效的智能体行为。
- **示例（Few-shot）**：始终推荐提供示例，但不建议把“所有边界条件”的罗列一股脑塞进提示。对 LLM 而言，**好的示例胜过千言万语**。
![Calibrating the system prompt](../docs/images/9-figures/9-2.webp)
<center>图 9.2 Calibrating the system prompt</center>

#### 9.2.2 上下文检索与智能体式搜索
一个简洁的定义：**智能体 = 在循环中自主调用工具的 LLM**。
- **及时（Just-in-time, JIT）上下文**
- **引用的元数据**
- **渐进式披露（progressive disclosure）**
- **混合策略**更有效

#### 9.2.3 面向长时程任务的上下文工程
指望无限增大上下文窗口并不能根治“上下文污染”与相关性退化的问题，因此需要直接面向这些约束的工程手段：**压缩整合（Compaction）**、**结构化笔记（Structured note-taking）** 与 **子代理架构（Sub-agent architectures）**。
- **压缩整合（Compaction）**: 适合需要长对话连续性的任务，强调上下文的“接力”。
- **结构化笔记（Structured note-taking）**: 适合有里程碑/阶段性成果的迭代式开发与研究。
- **子代理架构（Sub-agent architectures）**: 适合复杂研究与分析，能从并行探索中获益。

### 9.3 在 Hello-Agents 中的实践：ContextBuilder
#### 9.3.1 设计动机与目标
- **统一入口**: 将"获取(Gather)- 选择(Select)- 结构化(Structure)- 压缩(Compress)"抽象为可复用流水线
- **稳定形态**: 输出固定骨架的上下文模板，便于调试、A/B 测试与评估。我们采用了分区组织的模板结构：
  * [Role & Policies]：明确 Agent 的角色定位和行为准则
  * [Task]：当前需要完成的具体任务
  * [State]：Agent 的当前状态和上下文信息
  * [Evidence]：从外部知识库检索的证据信息
  * [Context]：历史对话和相关记忆
  * [Output]：期望的输出格式和要求
- **预算守护**
- **最小规则**

#### 9.3.2 核心数据结构
(1) ContextPacket：候选信息包
(2) ContextConfig：配置管理

#### 9.3.3 GSSC 流水线详解
ContextBuilder 的核心是 GSSC(Gather-Select-Structure-Compress)流水线

#### 9.3.4 完整使用示例

#### 9.3.5 最佳实践与优化建议
在实际应用 ContextBuilder 时，以下几点最佳实践值得注意：
- **动态调整 token 预算**
- **相关性计算优化**
- **缓存机制**
- **监控与日志**
- **A/B 测试**

### 9.4 NoteTool：结构化笔记
#### 9.4.1 设计理念与应用场景
(1) 为什么需要 NoteTool?
对于需要长期追踪、结构化管理的**项目式任务**，我们需要一种更轻量、更人类友好的记录方式。
NoteTool 提供了：
- **结构化记录**：使用 Markdown + YAML 格式，既适合机器解析，也方便人类阅读和编辑
- **版本友好**：纯文本格式，天然支持 Git 等版本控制系统
- **低开销**：无需复杂的数据库操作，适合轻量级的状态追踪
- **灵活分类**：通过 type 和 tags 灵活组织笔记，支持多维度检索

(2) 典型应用场景
- **场景1：长期项目追踪**
- **场景2：研究任务管理**
- **场景3：与 ContextBuilder 配合**

#### 9.4.2 存储格式详解
NoteTool 采用了 Markdown + YAML 的混合格式
(1) 笔记文件格式: 每个笔记都是一个独立的 `.md` 文件, 这种格式的优势：
  - **YAML 元数据**：机器可解析，支持精确的字段提取和检索
  - **Markdown 正文**：人类可读，支持丰富的格式化(标题、列表、代码块等)
  - **文件名即 ID**：简化管理，每个笔记的文件名就是其唯一标识
(2) 索引文件: NoteTool 维护一个 notes_index.json 文件，用于快速检索和管理笔记, 这个索引文件的作用：
  - **快速检索**：无需打开每个文件，直接从索引中查找
  - **元数据管理**：集中管理所有笔记的元数据
  - **完整性校验**：可以检测文件缺失或损坏

#### 9.4.3 核心操作详解
NoteTool 提供了七个核心操作，覆盖了笔记的完整生命周期管理: create, read, update, search, list, summary, delete

#### 9.4.4 与 ContextBuilder 的深度集成
NoteTool 的真正威力在于与 ContextBuilder 的配合使用。

#### 9.4.5 最佳实践
在实际使用 NoteTool 时，以下最佳实践能帮助您构建更强大的长时程智能体：
- **合理的笔记分类**
- **定期清理和归档**
- **与 ContextBuilder 的配合**
- **人机协作**
- **自动化工作流**

### 9.5 TerminalTool：即时文件系统访问
#### 9.5.1 设计理念与安全机制
(1) 为什么需要 TerminalTool?
  - **场景1：代码库探索**
  - **场景2：日志文件分析**
  - **场景3：数据文件预览**
这些场景的共同特点是：**需要实时、轻量级的文件系统访问，而不是预先索引和向量化**。TerminalTool 正是为这种"探索式"工作流设计的。

(2) 安全机制详解
TerminalTool 通过多层安全机制确保系统安全：
  - **第一层：命令白名单**
  - **第二层：工作目录限制(沙箱)**
  - **第三层：超时控制**
  - **第四层：输出大小限制**

#### 9.5.2 核心功能详解

#### 9.5.3 典型使用模式
(1) 探索式导航
(2) 数据文件分析
(3) 日志文件分析
(4) 代码库分析

#### 9.5.4 与其他工具的协同
TerminalTool 的真正威力在于与 MemoryTool、NoteTool 和 ContextBuilder 的协同使用。

### 9.6 长程智能体实战：代码库维护助手

#### 9.6.1 场景设定与需求分析

#### 9.6.2 系统架构设计
![图 9.3 代码库维护助手三层架构](../docs/images/9-figures/9-3.png)
<center>图 9.3 代码库维护助手三层架构</center>

#### 9.6.3 核心实现

#### 9.6.4 完整使用示例

#### 9.6.5 运行效果分析

### 9.7 本章总结
**理论层面**
- **上下文工程的本质**：从"提示工程"到"上下文工程"的演进，核心是管理有限的注意力预算
- **上下文腐蚀**：理解长上下文带来的性能下降，认识到上下文是稀缺资源
- **三大策略**：压缩整合、结构化笔记、子代理架构

**工程实践**
- **ContextBuilder**：实现了 GSSC 流水线，提供统一的上下文管理接口
- **NoteTool**：Markdown+YAML 的混合格式，支持结构化的长期记忆
- **TerminalTool**：安全的命令行工具，支持即时的文件系统访问
- **长程智能体**：整合三大工具，构建了跨会话的代码库维护助手

**核心收获**
- **分层设计**：即时访问(TerminalTool) + 会话记忆(MemoryTool) + 持久笔记(NoteTool)
- **智能筛选**：基于相关性和新近性的评分机制
- **安全第一**：多层安全机制确保系统稳定
- **人机协作**：自动化与可控性的平衡

### TODO: 例子程序运行问题与排错
chapter9 中的例子在运行过程中碰到一些问题, 调试和排错 ✅️⚠️
- [01_context_builder_basic.py](../code/chapter9/01_context_builder_basic.py): ContextBuilder 基础使用示例
- [02_context_builder_with_agent.py](../code/chapter9/02_context_builder_with_agent.py): ContextBuilder 与 Agent 集成示例
- [03_note_tool_operations.py](../code/chapter9/03_note_tool_operations.py): NoteTool 基本操作示例
- [04_note_tool_integration.py](../code/chapter9/04_note_tool_integration.py): NoteTool 与 ContextBuilder 集成示例
- [05_terminal_tool_examples.py](../code/chapter9/05_terminal_tool_examples.py): TerminalTool 使用示例
- [06_three_day_workflow.py](../code/chapter9/06_three_day_workflow.py): CodebaseMaintainer 三天工作流演示


## [第十章 智能体通信协议](https://datawhalechina.github.io/hello-agents/#/./chapter10/第十章%20智能体通信协议)
### 10.1 智能体通信协议基础
#### 10.1.1 为何需要通信协议
#### 10.1.2 三种协议设计理念比较
(1) MCP (Model Context Protocol, by Anthropic)：智能体与工具的桥梁
核心设计理念是标准化智能体与外部工具/资源的通信方式。MCP 的设计哲学是"上下文共享"。
![MCP 设计思想](../docs/images/10-figures/10-1.png)
<center>图 10.1 MCP 设计思想</center>

(2) A2A (Agent-to-Agent Protocol, by Google)：智能体间的对话
核心设计理念是**实现智能体之间的点对点通信**。A2A 的设计哲学是"对等通信"。与 MCP 关注智能体与工具的通信不同，A2A 关注的是智能体之间如何相互协作。
![A2A 设计思想](../docs/images/10-figures/10-2.png)
<center>图 10.2 A2A 设计思想</center>

(3) ANP (Agent Network Protocol, concept by Open Source Communities)：智能体网络的基础设施
核心设计理念是**构建大规模智能体网络的基础设施**。MCP 解决的是"如何访问工具"，A2A 解决的是"如何与其他智能体对话"，那么 ANP 解决的是"如何在大规模网络中发现和连接智能体"。设计哲学是"去中心化服务发现"。

![ANP 设计思想](../docs/images/10-figures/10-3.png)
<center>图 10.3 ANP 设计思想</center>


<center>表 10.1 三种协议对比</center>

| 维度 | MCP | A2A | ANP |
|---|---|---|---|
| 设计目标 | 智能体与工具 / 资源的标准化通信 | 智能体间的点对点通信 | 大规模智能体网络的服务发现 |
| 通信模式 | 客户端-服务器（C/S） | 对等网络（P2P） | 对等网络（P2P） |
| 核心理念 | 上下文共享 | 对等协作 | 去中心化发现 |
| 适用场景 | 访问外部工具和数据源 | 智能体协作和任务委托 | 大规模智能体生态系统 |
| 扩展性 | 通过添加 MCP 服务器扩展 | 通过添加智能体节点扩展 | 支持动态扩展 |
| 实现状态 | 已有成熟实现（FastMCP） | 官方 SDK 可用 | 概念性框架 |

(4) 如何选择合适的协议？
关键在于理解你的需求：
  - 如果你的智能体需要访问外部服务（文件、数据库、API），选择MCP
  - 如果你需要多个智能体相互协作完成任务，选择A2A
  - 如果你要构建大规模的智能体生态系统，考虑ANP

#### 10.1.3 HelloAgents 通信协议架构设计
HelloAgents 框架的设计目标是：**让学习者能够以最简单的方式使用这些协议，同时保持足够的灵活性以应对复杂场景。**
HelloAgents 的通信协议架构采用三层设计，从底层到上层分别是：协议实现层、工具封装层和智能体集成层。
![HelloAgents 通信协议设计](../docs/images/10-figures/10-4.png)
<center>图 10.4 HelloAgents 通信协议设计</center>

#### 10.1.4 本章学习目标与快速体验
```
hello_agents/
├── protocols/                          # 通信协议模块
│   ├── mcp/                            # MCP协议实现（Model Context Protocol）
│   │   ├── client.py                   # MCP客户端（支持5种传输方式）
│   │   ├── server.py                   # MCP服务器（FastMCP封装）
│   │   └── utils.py                    # 工具函数（create_context/parse_context）
│   ├── a2a/                            # A2A协议实现（Agent-to-Agent Protocol）
│   │   └── implementation.py           # A2A服务器/客户端（基于a2a-sdk，可选依赖）
│   └── anp/                            # ANP协议实现（Agent Network Protocol）
│       └── implementation.py           # ANP服务发现/注册（概念性实现）
└── tools/builtin/                      # 内置工具模块
    └── protocol_tools.py               # 协议工具包装器（MCPTool/A2ATool/ANPTool）
```
- [01_TestConnect.py](../code/chapter10/01_TestConnect.py): 简单的示例展示了三种协议的核心功能。

### 10.2 MCP 协议实战
#### 10.2.1 MCP 协议概念介绍
(1) MCP：智能体的"USB-C"
(2) MCP 架构: MCP 协议采用 Host、Client、Servers 三层架构设计
![MCP 案例演示](../docs/images/10-figures/10-5.png)
<center>图 10.5 MCP 案例演示</center>

**三层架构的职责：**
- **Host（宿主层）**：Claude Desktop 作为 Host，负责接收用户提问并与 Claude 模型交互。Host 是用户直接交互的界面，它管理整个对话流程。
- **Client（客户端层）**：当 Claude 模型决定需要访问文件系统时，Host 中内置的 MCP Client 被激活。Client 负责与适当的 MCP Server 建立连接，发送请求并接收响应。
- **Server（服务器层）**：文件系统 MCP Server 被调用，执行实际的文件扫描操作，访问桌面目录，并返回找到的文档列表。
**完整的交互流程**：用户问题 → Claude Desktop(Host) → Claude 模型分析 → 需要文件信息 → MCP Client 连接 → 文件系统 MCP Server → 执行操作 → 返回结果 → Claude 生成回答 → 显示在 Claude Desktop 上

(3) MCP 的核心能力
<center>表 10.2 MCP 核心能力</center>

| 能力 | 说明 | 使用场景 | 示例 |
|---|---|---|---|
| Tools（工具） | 可执行的功能，类似函数调用 | 执行操作、处理数据 | read_file，search_code，send_email |
| Resources（资源） | 可访问的数据，类似文件系统 | 读取数据、订阅变化 | 文件内容、数据库记录、API 响应 |
| Prompts（提示） | 预定义的提示模板 | 标准化任务、最佳实践 | 代码审查提示、文档生成提示 |

三种能力的区别在于：**Tools 是主动的**（执行操作），**Resources 是被动的**（提供数据），**Prompts 是指导性的**（提供模板）。

(4) MCP 的工作流程
![MCP 案例演示](../docs/images/10-figures/10-6.png)
<center>图 10.6 MCP 案例演示</center>

Claude（或其他 LLM）是如何决定使用哪些工具的？
- **工具发现阶段**: MCP Client 连接到 Server 后，首先调用`list_tools()`获取所有可用工具的描述信息（包括工具名称、功能说明、参数定义）
- **上下文构建**: Client 将工具列表转换为 LLM 能理解的格式，添加到系统提示词中。
- **模型推理**: LLM 分析用户问题和可用工具，决定是否需要调用工具以及调用哪个工具。
- **工具执行**: 如果 LLM 决定使用工具，Client 通过 MCP Server 执行所选工具，获取结果
- **结果整合**: 工具执行结果被送回给 LLM，LLM 结合结果生成最终回答
这个过程是**完全自动化**的，LLM 会根据工具描述的质量来决定是否使用以及如何使用工具。因此，编写清晰、准确的工具描述至关重要。

(5) MCP 与 Function Calling 的差异
<center>表 10.3 Function Calling 与 MCP 对比</center>

| 维度 | Function Calling | MCP |
|---|---|---|
| 本质 | LLM 的一种能力 | 标准化的通信协议 |
| 作用层级 | 模型层 | 基础设施层 |
| 解决问题 | 让 LLM 知道“如何调用函数” | 让工具和模型“如何连接” |
| 标准化 | 每个模型提供商实现不同 | 统一的协议规范 |
| 工具复用 | 需要为每个应用重写 | 社区工具可直接使用 |

#### 10.2.2 使用 MCP 客户端
(1) 连接到 MCP 服务器
(2) 发现可用工具
(3) 调用工具
(4) 访问资源
(5) 使用提示模板
(6) 完整示例：使用 GitHub MCP 服务

- [02_Connect2MCP.py](../code/chapter10/02_Connect2MCP.py): 使用 MCPClient 连接社区或者自定义的 MCP服务 以及通过 MCPClient 发现并使用工具 - connect_to_server, list_tools, list_resources, list_prompts, call_tool...
- [03_GitHubMCP.py](../code/chapter10/03_GitHubMCP.py): 完整示例：使用 GitHub MCP 服务

#### 10.2.3 MCP 传输方式详解
MCP 协议的一个重要特性是**传输层无关性**（Transport Agnostic）。这意味着 MCP 协议本身不依赖于特定的传输方式，可以在不同的通信通道上运行。HelloAgents 基于 FastMCP 2.0，提供了完整的传输方式支持，让你可以根据实际场景选择最合适的传输模式。
(1) 传输方式概览
HelloAgents的MCPClient支持五种传输方式
<center>表 10.4 MCP 传输方式对比</center>

| 传输方式 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| Memory | 单元测试、快速原型 | 最快、无网络开销 | 仅限同进程 |
| Stdio | 本地开发、命令行工具 | 简单、无需网络配置 | 仅限本地、可能有兼容性问题 |
| HTTP | 生产环境、远程服务 | 通用、防火墙友好 | 无流式支持、延迟较高 |
| SSE | 实时通信、流式响应 | 支持服务器推送 | 单向通信、需要 HTTP 服务器 |
| StreamableHTTP | 流式 HTTP 通信 | 双向流式、HTTP 兼容 | 需要特定服务器支持 |

(2) 传输方式使用示例
(3) Memory Transport - 内存传输
(4) Stdio Transport - 标准输入输出传输
(5) HTTP Transport - HTTP 传输
(6) SSE Transport - Server-Sent Events 传输
(7) StreamableHTTP Transport - 流式 HTTP 传输

- [04_MCPTransport.py](../code/chapter10/04_MCPTransport.py): 演示HelloAgents的MCPClient支持五种传输方式
- [server.py](./server.py): MCP Server 以 HTTP 传输方式启动

#### 10.2.4 在智能体中使用 MCP 工具
(1) MCP 工具的自动展开机制
(2) MCP 工具自动展开的工作原理
(3) 实战案例：智能文档助手

- [05_UseMCPToolInAgent.py](../code/chapter10/05_UseMCPToolInAgent.py): 演示MCP 工具的自动展开和通过Agent 调用
- [06_MultiAgentDocumentAssist.py](../code/chapter10/06_MultiAgentDocumentAssist.py): 多Agent协作的智能文档助手, 搜索github repos 并生成报告

#### 10.2.5 MCP 社区生态
MCP 社区的三个资源库：
- **[Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers)**
- **[MCP Servers Website](https://mcpservers.org/)**
- **[Official MCP Servers](https://github.com/modelcontextprotocol/servers)**
<center>表 10.5 常用官方 MCP 服务器</center>

| 服务器名称       | 功能              | NPM 包名                                              | 使用场景                     |
|------------------|-------------------|------------------------------------------------------|------------------------------|
| filesystem       | 文件系统访问      | @modelcontextprotocol/server-filesystem              | 读写本地文件、目录操作       |
| github           | GitHub API        | @modelcontextprotocol/server-github                  | 搜索仓库、读取代码           |
| postgres         | PostgreSQL 数据库 | @modelcontextprotocol/server-postgres                | 数据库查询、数据分析         |
| sqlite           | SQLite 数据库     | @modelcontextprotocol/server-sqlite                  | 轻量级数据库操作             |
| slack            | Slack 消息        | @modelcontextprotocol/server-slack                   | 发送消息、读取频道           |
| google-drive     | Google Drive      | @modelcontextprotocol/server-google-drive            | 访问云端文件                 |
| brave-search     | Brave 搜索        | @modelcontextprotocol/server-brave-search             | 网页搜索、实时信息获取       |
| fetch            | 网页抓取          | @modelcontextprotocol/server-fetch                   | 获取网页内容、提取数据       |

<center>表 10.6 社区热门 MCP 服务器</center>

| 服务器名称        | 功能         | 包名 / 仓库                     | 炫酷特性                                   |
|------------------|--------------|----------------------------------|--------------------------------------------|
| Playwright       | 浏览器自动化 | @playwright/mcp                 | 自动化网页交互、截图、填表单               |
| Puppeteer        | 浏览器控制   | mcp-server-puppeteer             | 网页爬取、PDF 生成                         |
| Screenpipe       | 屏幕录制     | mediar-ai/screenpipe             | 本地屏幕/音频捕获、时间戳索引、语义搜索    |
| Obsidian         | 笔记管理     | calclavia/mcp-obsidian           | 读取和搜索 Markdown 笔记、知识库管理       |
| Notion           | 协作文档     | Badhansen/notion-mcp             | 管理待办事项、数据库操作                   |
| Jira             | 项目管理     | nguyenvanduocit/jira-mcp         | Issue 管理、Sprint 规划、工作流            |
| Tavily           | AI 搜索      | kshern/mcp-tavily                | 专为 AI 优化的搜索 API                     |
| YouTube          | 视频处理     | anaisbetts/mcp-youtube           | 获取字幕、视频信息、转录内容               |
| Spotify          | 音乐控制     | marcelmarais/Spotify             | 播放控制、播放列表管理                     |
| Wolfram Alpha    | 计算知识     | ricocf/mcp-wolframalpha          | 数学计算、科学数据、实时知识               |
| Sentry           | 错误追踪     | getsentry/sentry-mcp             | 错误监控、性能分析                         |
| Grafana          | 可视化监控   | grafana/mcp-grafana              | 查询仪表板、数据源查询                     |


### 10.3 A2A 协议实战
#### 10.3.1 协议设计动机
传统的中央协调器（星型拓扑）方案存在三个主要问题：
- **单点故障**：协调器失效导致系统整体瘫痪。
- **性能瓶颈**：所有通信都经过中心节点，限制了并发。
- **扩展困难**：增加或修改智能体需要改动中心逻辑。

它的核心是 **任务（Task）** 和 **工件（Artifact）** 这两个抽象概念，这是它与 MCP 最大的区别

<center>表 10.7 A2A 核心概念</center>

| 概念            | 说明                     | 与 MCP 的区别               | 示例                         |
|-----------------|--------------------------|-----------------------------|------------------------------|
| Task（任务）     | 智能体之间委托的单元     | 比 Tool 更高层次的抽象      | “撰写一篇关于 AI 的文章”     |
| Artifact（工件） | 任务执行产生的结果       | 比 Resource 更结构化        | 文章文本、分析报告           |
| Message（消息）  | 智能体间的通信载体       | 包含任务状态信息            | “任务已完成 50%”             |
| Part（部分）     | 消息的组成部分           | 支持多模态内容              | 文本、图片、文件             |
| Agent Card      | 智能体描述文档           | 类似 MCP 的工具描述         | JSON 格式的能力声明          |

A2A 为任务定义了标准化的生命周期，包括创建、协商、代理、执行中、完成、失败等状态

![A2A 任务周期](../docs/images/10-figures/10-7.png)
<center>图 10.7 A2A 任务周期</center>

A2A 请求生命周期是一个序列，详细说明了请求遵循的四个主要步骤：代理发现、身份验证、发送消息 API 和发送消息流 API。

![A2A 请求生命周期](../docs/images/10-figures/10-8.png)
<center>图 10.8 A2A 请求生命周期</center>

#### 10.3.2 使用 A2A 协议实战
(1) 创建简单的 A2A 智能体
(2) 自定义 A2A 智能体

- [07_SimpleA2AAgent.py](../code/chapter10/07_SimpleA2AAgent.py): 演示简单的 A2A 智能体
- [08_CustomA2AAgent.py](../code/chapter10/08_CustomA2AAgent.py): 演示自定义 A2A 智能体

#### 10.3.3 使用 HelloAgents A2A 工具
(1) 创建 A2A Agent 服务端
(2) 创建 A2A Agent 客户端
(3) 创建 Agent 网络

- [09_A2A_Server.py](../code/chapter10/09_A2A_Server.py): 创建 A2A Agent 服务端
- [09_A2A_Client.py](../code/chapter10/09_A2A_Client.py): 创建 A2A Agent 客户端
- [09_A2A_Network.py](../code/chapter10/09_A2A_Network.py): 创建 Agent 网络

#### 10.3.4 在智能体中使用 A2A 工具
(1) 使用 A2ATool 包装器
(2) 实战案例：智能客服系统, 包含三个 Agent：**接待员**：分析客户问题类型; **技术专家**：回答技术问题; **销售顾问**：回答销售问题
(3) 高级用法：Agent 间协商

- [09_A2A_WithAgent.py](../code/chapter10/09_A2A_WithAgent.py): A2A 协议 + HelloAgents SimpleAgent 集成案例
- [10_A2ATool_Simple.py](../code/chapter10/10_A2ATool_Simple.py): 10.3.4 (1) 使用 A2ATool 包装器
- [10_AgentNegotiation.py](../code/chapter10/10_AgentNegotiation.py): 10.3.4 (3)高级用法 - Agent间协商
- [10_CustomerService.py](../code/chapter10/10_CustomerService.py): 10.3.4 (2)智能客服系统

### 10.4 ANP 协议实战
ANP（Agent Network Protocol）协议，专注于构建**大规模、开放的智能体网络**。

### 10.4.1 协议目标
当一个网络中存在大量功能各异的智能体（例如，自然语言处理、图像识别、数据分析等）时，系统会面临一系列挑战：
- **服务发现**：当新任务到达时，如何快速找到能够处理该任务的智能体？
- **智能路由**：如果多个智能体都能处理同一任务，如何选择最合适的一个（如根据负载、成本等）并向其分派任务？
- **动态扩展**：如何让新加入网络的智能体被其他成员发现和调用？

<center>表 10.8 ANP 核心概念</center>

| 概念              | 说明                                                                 | 示例                                             |
|-------------------|----------------------------------------------------------------------|--------------------------------------------------|
| ANP Discovery     | 服务发现中心，用于注册和查询网络中的智能体服务。                     | 一个中央服务器或一个 P2P 的 DHT 网络。           |
| Service Info      | 描述智能体服务的信息，包括其能力、地址和元数据。                     | {"agent_id": "nlp-agent-01", ...}                |
| ANP Network       | 对智能体网络的抽象，管理节点间的连接与通信。                         | 整个智能体集群的拓扑视图。                       |
| Capability        | 描述智能体功能的能力标签，用于服务发现时的匹配。                     | "text_analysis"、"image_processing"              |
| Metadata          | 服务的动态或静态元数据，用于路由决策。                               | 负载情况、服务价格、软件版本等。                 |

![ANP 整体流程](../docs/images/10-figures/10-9.png)
<center>图 10.9 ANP 整体流程</center>

主要包括以下几个步骤：
- **服务的发现与匹配**
- **基于 DID 的身份验证**
- **标准化的服务执行**

### 10.4.2 使用 ANP 服务发现
- [11_ANPInit.py](../code/chapter10/11_ANPInit.py): 10.4.2 使用 ANP 服务发现

### 10.4.3 实战案例
- [12_ANPTaskDistribution.py](../code/chapter10/12_ANPTaskDistribution.py): 10.4.3 实战案例 - 分布式任务调度系统
- [13_ANPLoadBalancing.py](../code/chapter10/13_ANPLoadBalancing.py): 10.4.3 实战案例 - 负载均衡示例

### 10.5 构建自定义 MCP 服务器
#### 10.5.1 创建你的第一个 MCP 服务器
(1) 为什么要构建自定义 MCP 服务器？
- **封装业务逻辑**
- **访问私有数据**
- **性能专项优化**
- **功能定制扩展**
(2) 教学案例：天气查询 MCP 服务器
(3) 测试自定义 MCP 服务器
(4) 在 Agent 中使用自定义 MCP 服务器

- [14_weather_mcp_server.py](../code/chapter10/14_weather_mcp_server.py): 天气查询 MCP 服务器
- [14_test_weather_server.py](../code/chapter10/14_test_weather_server.py): 测试自定义 MCP 服务器
- [14_weather_agent.py](../code/chapter10/14_weather_agent.py): 在 Agent 中使用自定义 MCP 服务器

#### 10.5.2 上传 MCP 服务器
(1) [Smithery](https://smithery.ai/) 是 MCP 服务器的官方发布平台，类似于 Python 的 PyPI 或 Node.js 的 npm。
(2) 需要将项目整理成标准的发布格式
```
weather-mcp-server/
├── README.md           # 项目说明文档
├── LICENSE            # 开源许可证
├── Dockerfile         # Docker 构建配置（推荐）
├── pyproject.toml     # Python 项目配置（必需）
├── requirements.txt   # Python 依赖
├── smithery.yaml      # Smithery 配置文件（必需）
└── server.py          # MCP 服务器主文件
```
(3) 提交到 Smithery

### 10.6 本章总结
三种核心协议：MCP、A2A 与 ANP 定位
- **MCP (Model Context Protocol)**: 作为智能体与工具之间的桥梁，提供统一的工具访问接口，适用于增强单个智能体的能力。
- **A2A (Agent-to-Agent Protocol)**: 作为智能体之间的对话系统，支持直接通信与任务协商，适用于小规模团队的紧密协作。
- **ANP (Agent Network Protocol)**: 作为智能体的“互联网”，提供服务发现、路由与负载均衡机制，适用于构建大规模、开放的智能体网络。

**深入学习**：
- 阅读 MCP 官方文档：https://modelcontextprotocol.io
- 阅读 A2A 官方文档：https://a2a-protocol.org/latest/
- 阅读 ANP 官方文档：https://agent-network-protocol.com/guide/

### TODO: 例子程序运行问题与排错
chapter10 中的例子在运行过程中碰到一些问题, 调试和排错 ✅️⚠️
- [01_TestConnect.py](../code/chapter10/01_TestConnect.py)
- [02_Connect2MCP.py](../code/chapter10/02_Connect2MCP.py)
- [03_GitHubMCP.py](../code/chapter10/03_GitHubMCP.py)
- [04_MCPTransport.py](../code/chapter10/04_MCPTransport.py)
- [05_UseMCPToolInAgent.py](../code/chapter10/05_UseMCPToolInAgent.py)
- [06_MultiAgentDocumentAssist.py](../code/chapter10/06_MultiAgentDocumentAssist.py)
- [07_SimpleA2AAgent.py](../code/chapter10/07_SimpleA2AAgent.py): 演示简单的 A2A 智能体
- [08_CustomA2AAgent.py](../code/chapter10/08_CustomA2AAgent.py): 演示自定义 A2A 智能体
- [09_A2A_Server.py](../code/chapter10/09_A2A_Server.py): 创建 A2A Agent 服务端
- [09_A2A_Client.py](../code/chapter10/09_A2A_Client.py): 创建 A2A Agent 客户端
- [09_A2A_Network.py](../code/chapter10/09_A2A_Network.py): 创建 Agent 网络
- [09_A2A_WithAgent.py](../code/chapter10/09_A2A_WithAgent.py): A2A 协议 + HelloAgents SimpleAgent 集成案例
- [10_A2ATool_Simple.py](../code/chapter10/10_A2ATool_Simple.py): 10.3.4 (1)使用A2ATool包装器
- [10_AgentNegotiation.py](../code/chapter10/10_AgentNegotiation.py): 10.3.4 (3)高级用法 - Agent间协商
- [10_CustomerService.py](../code/chapter10/10_CustomerService.py): 10.3.4 (2)智能客服系统
- [11_ANPInit.py](../code/chapter10/11_ANPInit.py): 10.4.2 使用 ANP 服务发现
- [12_ANPTaskDistribution.py](../code/chapter10/12_ANPTaskDistribution.py): 10.4.3 实战案例 - 分布式任务调度系统
- [13_ANPLoadBalancing.py](../code/chapter10/13_ANPLoadBalancing.py): 10.4.3 实战案例 - 负载均衡示例
- [14_weather_mcp_server.py](../code/chapter10/14_weather_mcp_server.py): 天气查询 MCP 服务器
- [14_test_weather_server.py](../code/chapter10/14_test_weather_server.py): 测试自定义 MCP 服务器
- [14_weather_agent.py](../code/chapter10/14_weather_agent.py): 在 Agent 中使用自定义 MCP 服务器


## [第十一章 Agentic-RL](https://datawhalechina.github.io/hello-agents/#/./chapter11/第十一章%20Agentic-RL)
### 11.1 从 LLM 训练到 Agentic RL
我们将从 LLM 训练的基础知识开始，逐步深入到 **SFT** (Supervised Fine-Tuning, 监督微调)、**GRPO** (Group Relative Policy Optimization, 群组相对策略优化)等实用技术，最终构建一个完整的智能体训练 pipeline。
**DPO** (Direct Preference Optimization)

#### 11.1.1 从强化学习到 Agentic RL
强化学习框架:
- **智能体**: 基于 LLM 的推理系统
- **环境**: 数学问题和验证系统
- **状态**: 当前的问题描述和已有的推理步骤
- **行动**: 生成下一步推理或最终答案
- **奖励**: 答案是否正确(正确+1，错误 0)

#### 11.1.2 LLM 训练全景图
一个强大的 LLM(如 GPT、Claude、Qwen)的诞生，通常要经历两个主要阶段:预训练(Pretraining)和后训练(Post-training)。
![LLM 训练全景图](../docs/images/11-figures/11-1.png)
<center>图 11.1 LLM 训练全景图</center>

**预训练阶段**是 LLM 训练的第一阶段，目标是让模型学习语言的基本规律和世界知识。最常见的预训练任务是因果语言建模(Causal Language Modeling)，也称为下一个词预测(Next Token Prediction)。
给定一个文本序列 $x_1, x_2, \ldots, x_t$，模型需要预测下一个词 $x_{t+1}$：

$$
\mathcal{L}_{\text{pretrain}}
= - \sum_{t=1}^{T} \log P\bigl(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta \bigr)
$$

其中：
- $\theta$ 是模型参数  
- $P(x_t \mid x_1, \ldots, x_{t-1}; \theta)$ 是模型预测的下一个词的概率分布  
- 目标是最小化负对数似然（Negative Log-Likelihood），即最大化预测正确词的概率

**后训练阶段**则是要解决预训练模型的不足。让模型对齐人类的偏好和价值观。

后训练通常包含三个步骤。
- 第一步是**监督微调（Supervised Fine-Tuning, SFT）**，目标是让模型学会遵循指令和对话格式。
训练数据是 $(\text{prompt}, \text{completion})$ 对，训练目标与预训练类似，仍然是最大化正确输出的概率：
$$
\mathcal{L}_{\mathrm{SFT}}
= - \sum_{i=1}^{N} \log P(y_i \mid x_i; \theta)
$$

其中：$x_i$ 是输入提示（prompt）; $y_i$ 是期望的输出; $N$ 是训练样本数量; $\theta$ 是模型参数。
**SFT 的特点**是数据量较小、需要人工标注、快速见效，主要学习任务格式和基本能力。

- 第二步是**奖励建模（Reward Modeling, RM）**。SFT 后的模型虽然能遵循指令，但生成的回答质量参差不齐。  
因此需要一种方式来评估回答质量，这就是奖励模型的作用。奖励模型的训练目标是学习人类的偏好：

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} [\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]
$$

其中：$r_\phi(x, y)$ 是奖励模型, 输入是 $(\text{提示}, \text{回答})$ 对, 输出是质量分数; 
$y_w$ 是更好的回答（chosen）, $y_l$ 是更差的回答（rejected）, $\sigma(\cdot)$ 是 sigmoid 函数, 目标是让奖励模型为更好的回答给出更高的分数。

- 第三步是**强化学习微调**。在有了奖励模型之后，可以使用强化学习来优化语言模型，使其生成更高质量的回答。最经典的算法是 **PPO（Proximal Policy Optimization）**。
训练目标为：

$$
J_{\text{PPO}} = \mathbb{E}_{x, y \sim \pi_\theta} [r_\phi(x, y)] - \beta \cdot D_{KL}(\pi_\theta || \pi_{\text{ref}})
$$

其中：$\pi_\theta$ 是当前策略（语言模型）, $\pi_{\mathrm{ref}}$ 是参考策略（通常是 SFT 模型）, $r_\phi(x, y)$ 是奖励模型给出的评分; 
$D_{\mathrm{KL}}(\cdot)$ 是 KL 散度, $\beta$ 是平衡系数。该目标函数的含义是：
**在最大化奖励的同时，防止模型偏离原始模型过远。**

传统的 RLHF(Reinforcement Learning from Human Feedback) 需要大量人工标注偏好数据，成本高昂。为了降低成本，研究者提出了 RLAIF(Reinforcement Learning from AI Feedback)，
用强大的 AI 模型(如 GPT-4)来替代人类标注员。RLAIF 的工作流程是:用 SFT 模型生成多个候选回答，用强大的 AI 模型对回答进行评分和排序，用 AI 的评分训练奖励模型，用奖励模型进行强化学习。
实验表明，RLAIF 的效果接近甚至超过 RLHF，同时成本大幅降低。

#### 11.1.3 Agentic RL 的核心理念
- 传统的后训练(我们称之为 PBRFT: Preference-Based Reinforcement Fine-Tuning)主要关注单轮对话的质量优化:给定一个用户问题，模型生成一个回答，然后根据回答的质量获得奖励。
- Agentic RL则是一种新的范式，它将 LLM 视为一个可学习的策略，嵌入在一个顺序决策循环中。在这个框架下，智能体需要在动态环境中与外部世界交互，执行多步行动来完成复杂任务，获得中间反馈来指导后续决策，优化长期累积奖励而非单步奖励。

强化学习是基于马尔可夫决策过程(Markov Decision Process， MDP)框架进行形式化的。MDP 由五元组 $(S, A, P, R, \gamma)$ 定义:状态空间S、行动空间A、状态转移函数P(s′∣s,a)、奖励函数R(s,a)、折扣因子γ。
<center>表 11.1 PBRFT 与 Agentic RL 对比</center>

| 维度 | PBRFT | Agentic RL |
|----|-------|------------|
| 状态 | 单一提示 $s_0$ | 动态演化 $s_t$ |
| 行动 | 文本生成 $a = y \sim \pi_\theta(y\|s_0)$  | 文本 + 工具 + 环境操作 $a_t \in \{a_t^{\text{text}}, a_t^{\text{tool}}\}$ |
| 转移 | 无转移 $P(s'\|s,a) = \delta(s' - s_{\text{terminal}})$ | 状态随行动变化 $s_{t+1} \sim P(s_{t+1}\|s_t, a_t)$ |
| 奖励 | 单步 $r(s_0, y)$ | 累积 $\sum_t \gamma^t r(s_t, a_t)$ |
| 时间 | $T = 1$ | $T \gg 1$ |
| 目标 | 短期质量 | 长期成功 |

![Agentic RL 的六大核心能力](../docs/images/11-figures/11-2.png)
<center>图 11.2 Agentic RL 的六大核心能力</center>

#### 11.1.4 HelloAgents 的 Agentic RL 设计
- **推理(Reasoning)**: 从给定信息中逻辑地得出结论的过程，是智能体的核心能力。CoT(Chain of Thoughts), SFT (Supervised Fine-Tuning), RL (Reinfoced Learning)
- **工具使用(Tool Use)**: 智能体调用外部工来完成任务的能力。
- **记忆(Memory)**: 智能体保持和重用过去信息的能力，对于长期任务至关重要。
- **规划(Planning)**: 制定行动序列以达成目标的能力。
- **自我改进(Self-Improvement)**: 智能体回顾自身输出、纠正错误并优化策略的能力。
- **感知(Perception)**: 理解多模态信息的能力。强化学习可以提升视觉推理能力，让模型学会使用视觉工具，学会视觉规划。这使得智能体不仅能理解文本，还能理解和操作视觉世界。

![HelloAgents Agentic RL 架构](../docs/images/11-figures/11-3.png)
<center>图 11.3 HelloAgents Agentic RL 架构</center>

HelloAgents 的 Agentic RL 模块采用四层架构设计: **数据集层** - **奖励函数层** - **训练器层** - **统一接口层**

#### 11.1.5 快速上手示例
```bash
pip install "hello-agents[rl]"
# 安装依赖包 trl
micromamba install trl
# 测试 transformers 依赖包的完整和一致
python -c "from transformers import AutoTokenizer; print('ok')"
# 如果碰到 import 错误, 需要重新安装某些依赖库, 最好用 micromamba 安装, 必要时需要彻底删除一些残余文件
python -m pip uninstall -y transformers tokenizers huggingface_hub safetensors
rm -rf ~/micromamba/envs/v3.12.12/lib/python3.12/site-packages/huggingface_hub*
micromamba install -c conda-forge transformers tokenizers huggingface_hub safetensors
python -m pip install --no-cache-dir -U huggingface_hub==0.36.0
```

### 11.2 数据集与奖励函数
数据集和奖励函数是强化学习训练的两大基石。数据集定义了智能体要学习的任务，奖励函数定义了什么是好的行为。

#### 11.2.1 GSM8K 数学推理数据集
GSM8K(Grade School Math 8K)是一个高质量的小学数学应用题数据集。
<center>表 11.2 GSM8K 数据集统计</center>

| 属性         | 值                     |
|--------------|------------------------|
| 训练集大小   | 7,473 个问题           |
| 测试集大小   | 1,319 个问题           |
| 难度等级     | 小学数学（2–8 年级）   |
| 题型         | 应用题                 |
| 推理步骤     | 2–8 步                 |
| 答案类型     | 数值                   |


![GSM8K 数据格式转换](../docs/images/11-figures/11-4.png)
<center>图 11.4 GSM8K 数据格式转换</center>

<center>表 11.3 数据格式对比</center>

| 格式 | 用途       | 标签内容           | 特点         |
|------|------------|--------------------|--------------|
| 原始 | 数据存储   | answer（含步骤）   | 人类可读     |
| SFT  | 监督学习   | 完整解答           | 学习格式     |
| RL   | 强化学习   | 仅答案             | 自主推理     |

#### 11.2.2 奖励函数设计
奖励函数是强化学习的核心，它定义了什么是"好的行为"。
在强化学习中，奖励函数 $r(s, a)$ 或 $r(s, a, s')$ 为智能体的每个行动分配一个数值奖励。智能体的目标是最大化累积奖励:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]
$$

对于数学推理任务，我们可以简化为:

$$
r(q, a) = f(a, a^*)
$$

其中 $q$ 是问题，$a$ 是模型生成的答案，$a^*$ 是正确答案，$f$ 是评估函数。

![奖励函数设计](../docs/images/11-figures/11-5.png)
<center>图 11.5 奖励函数设计</center>

(1) **准确率奖励**
准确率奖励(AccuracyReward)是最基础的奖励函数，它只关心答案是否正确。数学定义为:

$$
r_{\text{acc}}(a, a^*) = \begin{cases}
1 & \text{if } a = a^* \\
0 & \text{otherwise}
\end{cases}
$$

其中 $a$ 是模型生成的答案，$a^*$ 是正确答案。

(2) **长度惩罚**
长度惩罚(LengthPenaltyReward)鼓励模型生成简洁的回答，避免冗长啰嗦。数学定义为:

$$
r_{\text{length}}(a, a^*, l) = r_{\text{acc}}(a, a^*) - \alpha \cdot \max(0, l - l_{\text{target}})
$$

其中 $l$ 是生成文本的长度(字符数或 token 数)，$l_{\text{target}}$ 是目标长度，$\alpha$ 是惩罚系数(默认 0.001)。只有在答案正确的情况下才应用长度惩罚，避免模型为了减少惩罚而生成错误的短答案。

(3) **步骤奖励**
步骤奖励(StepReward)鼓励模型生成清晰的推理步骤，提高可解释性。数学定义为:

$$
r_{\text{step}}(a, a^*, s) = r_{\text{acc}}(a, a^*) + \beta \cdot s
$$

其中 $s$ 是检测到的推理步骤数量，$\beta$ 是步骤奖励系数(默认 0.1)。同样，只有在答案正确的情况下才给予步骤奖励。

在实际应用中，我们通常会组合多个奖励函数，以平衡不同的目标。常见的组合策略包括:

<strong>准确率 + 长度惩罚</strong>:鼓励简洁正确的答案，适合对话系统、问答系统。公式为:

$$
r = r_{\text{acc}} - \alpha \cdot \max(0, l - l_{\text{target}})
$$

<strong>准确率 + 步骤奖励</strong>:鼓励详细的推理过程，适合教育场景、可解释 AI。公式为:

$$
r = r_{\text{acc}} + \beta \cdot s
$$

<strong>三者平衡</strong>:全面优化答案质量、简洁性和可解释性。公式为:
$$
r = r_{\text{acc}} - \alpha \cdot \max(0, l - l_{\text{target}}) + \beta \cdot s
$$

需要仔细调整权重 $\alpha$ 和 $\beta$，避免某个目标过度主导。

<center>表 11.4 奖励函数对比</center>

| 奖励函数   | 优点         | 缺点           | 适用场景   |
|------------|--------------|----------------|------------|
| 准确率     | 简单直接     | 奖励稀疏       | 基础训练   |
| 长度惩罚   | 鼓励简洁     | 可能抑制推理   | 对话系统   |
| 步骤奖励   | 可解释性强   | 可能冗余       | 教育应用   |
| 组合奖励   | 全面优化     | 调参复杂       | 生产环境   |

#### 11.2.3 自定义数据集和奖励函数
- **SFT 格式**: 用于监督微调, 需要包含以下字段: `prompt`, `completion`, `text`
- **RL 格式**: 用于强化学习，需要包含以下字段: `question`, `prompt`, `ground_truth`, `full_answer`

(1) **使用 format_math_dataset 转换**
(2) **直接传入自定义数据集**
(3) **注册自定义数据集(推荐)**

自定义奖励函数:
```python
def custom_reward_function(
    completions: List[str],
    **kwargs
) -> List[float]:
```

使用自定义奖励函数: (1) 直接传入; (2) 注册使用(推荐)

### 11.3 SFT 训练
#### 11.3.1 为什么需要 SFT
SFT 的作用是教会模型任务的基本规则。
- 首先，学习输出格式，让模型知道如何组织答案(如使用"Step 1"， "Final Answer"等标记)。
- 其次，学习推理模式，通过示例学习如何分解问题、逐步推导。
- 再次，建立基线能力，为后续的强化学习提供一个合理的起点。
- 最后，减少探索空间，强化学习不需要从零开始，可以在 SFT 的基础上优化。

SFT 是从预训练模型到强化学习的桥梁。
![SFT 在训练流程中的作用](../docs/images/11-figures/11-6.png)
<center>图 11.6 SFT 在训练流程中的作用</center>

#### 11.3.2 LoRA:参数高效微调
**LoRA(Low-Rank Adaptation)** 是一种参数高效微调方法，它只训练少量的额外参数，而保持原模型参数冻结。LoRA 的核心思想是:模型微调时的参数变化可以用低秩矩阵表示。
<center>表 11.5 LoRA vs 全量微调对比</center>

| 模型        | 全量参数 | LoRA 参数 (r=8) | 显存（全量） | 显存（LoRA） |
|-------------|----------|------------------|--------------|--------------|
| Qwen3-0.6B  | 0.6B     | 2.4M             | 12GB         | 4GB          |
| Qwen3-1.5B  | 1.5B     | 6.0M             | 24GB         | 8GB          |
| Qwen3-7B    | 7B       | 28M              | 112GB        | 28GB         |

LoRA 的关键超参数包括:
- r (rank, 秩): 控制 LoRA 矩阵的秩，越大表达能力越强，但参数量也越多，典型值为 4-64，默认 8
- α (alpha): LoRA 的缩放因子，实际更新为 $\Delta W = \frac{\alpha}{r} BA$，控制 LoRA 的影响强度，典型值等于 rank
- target_modules (目标模块): 指定哪些层应用 LoRA，通常选择注意力层(q_proj， k_proj， v_proj， o_proj)，也可以包括 MLP 层(gate_proj， up_proj， down_proj)

#### 11.3.3 SFT 训练实战
(1) **训练参数详解**: 数据参数; 训练参数; LoRA 参数; 优化器参数
(2) **完整训练示例**
(3) **训练监控和调试**
在训练过程中，我们需要监控三个关键指标: 
- **Loss(损失)**: 应该逐渐下降，如果不下降可能是学习率太小或数据有问题，如果下降后又上升则可能是学习率太大或出现过拟合
- **Gradient Norm(梯度范数)**: 应该在 0.1-10 的合理范围内，过大(>100)说明出现梯度爆炸需要降低学习率，过小(<0.01)说明梯度消失需要检查模型配置。
- **Learning Rate(学习率)**: 应该按照 warmup 策略变化，前 10%步数线性增加，然后线性衰减到 0。

训练中常见的问题及解决方案:
- 显存不足时可以减小 batch_size 或 max_length，使用梯度累积或更小的模型
- 训练速度慢时可以增大 batch_size，减少 logging 频率，或使用混合精度训练
- 损失不下降时可以增大学习率，检查数据格式，或增加训练轮数
- 过拟合时可以增大 weight_decay，减少训练轮数，或使用更多数据

#### 11.3.4 模型评估
训练完成后，我们需要评估模型的效果。评估指标包括:
- **准确率(Accuracy)**: 答案完全正确的比例，最直接的指标，范围 0-1，越高越好。
- **平均奖励(Average Reward)**: 所有样本的平均奖励，综合考虑准确率、长度、步骤等因素，范围取决于奖励函数设计。
- **推理质量(Reasoning Quality)**: 推理过程的清晰度和逻辑性，需要人工评估或使用专门的评估模型。

<center>模型准确率对比</center>

| 模型                | LoRA   | 预训练参数       | 准确度    |
|--------------------|--------|-----------------|----------|
| Qwen/Qwen3-0.6B    | false  | N/A             |   3.00%  |
| sft-full           |  true  | r=16, alpha=32  |  28.00%  |
| sft-full2          |  true  | r=16, alpha=32  |  39.00%  |

sft-full2: "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]

# 预训练模型准确率: 3.00%
# SFT模型准确率: 28.00%

### 11.4 GRPO 训练
#### 11.4.1 从 PPO 到 GRPO
在强化学习领域，**PPO(Proximal Policy Optimization)** 是最经典的算法之一。PPO 通过限制策略更新的幅度，保证训练的稳定性。但是，PPO 在 LLM 训练中存在一些问题: 需要训练 Value Model(价值模型)，增加了训练复杂度和显存占用; 需要同时维护四个模型(Policy Model、Reference Model、Value Model、Reward Model)，工程实现复杂; 训练不稳定，容易出现奖励崩塌或策略退化。

**GRPO(Group Relative Policy Optimization)** 是一种简化的 PPO 变体，专门为 LLM 设计。GRPO 的核心思想是: 不需要 Value Model，使用组内相对奖励代替绝对奖励; 简化训练流程，只需要 Policy Model 和 Reference Model; 提高训练稳定性，减少奖励崩塌的风险。

![PPO vs GRPO 训练流程](../docs/images/11-figures/11-7.png)
<center>图 11.7 PPO vs GRPO 训练流程</center>

<center>表 11.6 PPO vs GRPO 对比</center>

| 维度       | PPO                                   | GRPO                 |
|------------|----------------------------------------|----------------------|
| 模型数量   | 4 个（Policy, Ref, Value, Reward）    | 2 个（Policy, Ref）  |
| 优势估计   | Value Model                            | 组内相对奖励         |
| 显存占用   | 高（需要 Value Model）                | 低（无 Value Model） |
| 训练稳定性 | 中等                                   | 较高                 |
| 实现复杂度 | 高                                     | 低                   |
| 适用场景   | 通用 RL                                | LLM 微调             |

#### 11.4.2 GRPO 训练实战
GRPO 训练的前提是已经完成 SFT 训练，因为 GRPO 需要一个合理的初始策略。如果 GRPO 训练过程中平均奖励逐渐提升，KL 散度保持在合理范围内，说明训练正常进行。
GRPO 有一些特定的参数需要理解和调优。
- **生成参数**:
  * `num_generations`: 每个问题生成多少个答案。越多越好，但计算成本也越高。典型值为 4-8。生成多个答案的目的是计算组内相对奖励，增加训练信号的多样性。
  * `max_new_tokens`: 每个答案最多生成多少个 token。太少可能截断答案，太多浪费计算。建议 256-512。
  * `temperature`: 生成温度，控制随机性。0 表示贪婪解码，1 表示标准采样。GRPO 建议 0.7-1.0，保持一定的探索性。

- **优化参数**:
  * `learning_rate`: GRPO 的学习率通常比 SFT 小，因为我们不想偏离 SFT 模型太远。建议 1e-5 到 5e-5。
  * `kl_coef`: KL 散度惩罚系数，控制策略更新的幅度。太小(0.01)可能导致策略偏离太远，太大(0.5)可能限制学习。建议 0.05-0.1。
  * `clip_range`: 策略比率裁剪范围，类似 PPO 的 epsilon。建议 0.2。

- **奖励参数**:
  * `reward_type`: 奖励函数类型，可以是"accuracy"、"length_penalty"、"step"或"combined"。
  * `reward_config`: 奖励函数的额外配置，如长度惩罚的目标长度、步骤奖励的系数等。

#### 11.4.3 GRPO 训练过程解析
(1) **训练循环**: 训练循环 -> 奖励计算 -> 相对奖励 -> 策略更新 -> 重复
相对奖励机制鼓励模型生成"比平均水平更好"的答案，而不是简单地追求高奖励。这样可以减少奖励方差，提高训练稳定性。

(2) **KL 散度惩罚**: KL 散度惩罚是 GRPO 的关键组成部分，它防止策略偏离参考模型太远。KL 散度定义为:

$$
D_{KL}(\pi_\theta || \pi_{\text{ref}}) = \mathbb{E}_{s,a \sim \pi_\theta} \left[ \log \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)} \right]
$$

在实践中，我们计算每个 token 的 KL 散度，然后求和:

$$
D_{KL} = \sum_{t=1}^{T} \log \frac{\pi_\theta(a_t|s, a_{<t})}{\pi_{\text{ref}}(a_t|s, a_{<t})}
$$

KL 散度越大，说明当前策略与参考模型差异越大。通过添加 KL 散度惩罚项 $-\beta \cdot D_{KL}$，我们限制策略更新的幅度，避免"遗忘"SFT 阶段学到的知识。

`kl_coef` ($\beta$) 的选择很重要:

- 太小(0.01):策略可能偏离太远，导致输出格式混乱或质量下降
- 太大(0.5):策略更新受限，学习缓慢，难以超越 SFT 模型
- 建议(0.05-0.1):平衡探索和稳定性

(3) **训练监控**: 在 GRPO 训练过程中，我们需要监控以下指标:
- **平均奖励(Average Reward)**
- **KL 散度(KL Divergence)**
- **准确率(Accuracy)**
- **生成质量(Generation Quality)**

HelloAgents 集成了两种主流的训练监控工具:Weights & Biases(wandb)和 TensorBoard。
(1) 方式 1: 使用 Weights & Biases(推荐)
(2) 方式 2: 使用 TensorBoard
(3) 方式 3: 离线监控(无需外部工具)

### 11.5 模型评估与分析
#### 11.5.1 评估指标体系
(1) **准确性指标**: 
  - **准确率(Accuracy)**: $$\text{Accuracy} = \frac{\text{正确答案数}}{\text{总问题数}}$$
  - **Top-K 准确率**: $$\text{Accuracy@K} = \frac{\text{至少有一个正确答案的问题数}}{\text{总问题数}}$$
  - **数值误差(Numerical Error)**: $$\text{Error} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

(2) **效率指标**:
  - **平均长度(Average Length)**: 生成答案的平均 token 数。计算公式为: $$\text{Avg Length} = \frac{1}{N} \sum_{i=1}^{N} |y_i|$$
  - **推理步骤数(Reasoning Steps)**: 答案中包含的推理步骤数量。计算公式为: $$\text{Avg Steps} = \frac{1}{N} \sum_{i=1}^{N} s_i$$
  - **推理时间(Inference Time)**: 生成一个答案所需的时间。

(3) **质量指标**:
  - **格式正确率(Format Correctness)**: 答案是否符合预期格式(如包含"Step 1"， "Final Answer"等标记)。计算公式为: $$\text{Format Correctness} = \frac{\text{格式正确的答案数}}{\text{总答案数}}$$
  - **推理连贯性(Reasoning Coherence)**: 推理步骤之间是否逻辑连贯。这个指标通常需要人工评估或使用专门的评估模型。
  - **可解释性(Explainability)**: 答案是否容易理解和验证。包含清晰步骤的答案比直接给出结果的答案更具可解释性。

<center>表 11.7 评估指标对比</center>
| 类别   | 指标                 | 优点         | 缺点           |
|--------|----------------------|--------------|----------------|
| 准确性 | Accuracy             | 简单直观     | 过于粗糙       |
| 准确性 | Accuracy@K           | 反映潜力     | 需要多次采样   |
| 准确性 | Numerical Error      | 细粒度       | 仅适用数值任务 |
| 效率   | Avg Length           | 反映成本     | 不考虑质量     |
| 效率   | Avg Steps            | 反映推理风格 | 难以量化       |
| 质量   | Format Correctness   | 易于检测     | 不保证正确性   |
| 质量   | Coherence            | 全面评估     | 需要人工       |

#### 11.5.2 评估实战

#### 11.5.3 错误分析

#### 11.5.4 改进方向
![模型改进迭代流程](../docs/images/11-figures/11-8.png)
<center>图 11.8 模型改进迭代流程</center>

### 11.6 完整训练流程实战

#### 11.6.1 端到端训练流程
![端到端训练流程](../docs/images/11-figures/)
<center>图 11.9 端到端训练流程</center>

运行小建议：
- **从小规模开始**
- **数据质量检查**
- **数据增强**

#### 11.6.2 超参数调优
(1) **网格搜索(Grid Search)**
(2) **随机搜索(Random Search)**
(3) **贝叶斯优化(Bayesian Optimization)**: 可以使用 Optuna 等库

<center>表 11.8 超参数调优方法对比</center>

| 方法       | 优点             | 缺点           | 适用场景            |
|------------|------------------|----------------|---------------------|
| 网格搜索   | 简单，全局最优   | 计算成本高     | 参数少（2–3 个）    |
| 随机搜索   | 效率高           | 可能错过最优   | 参数多（4–6 个）    |
| 贝叶斯优化 | 样本效率高       | 实现复杂       | 计算资源有限        |

#### 11.6.3 分布式训练
方案选择建议:
- **单机多卡(2-8 卡)**: 使用 DDP，简单高效
- **大模型(>7B)**: 使用 DeepSpeed ZeRO-2 或 ZeRO-3
- **多节点集群**: 使用 DeepSpeed ZeRO-3 + Offload

(1) 配置 Accelerate
(2) 使用 DDP 训练
(3) 使用 DeepSpeed ZeRO 训练
<center>表 11.9 显存对比 (Qwen3-0.6B 模型)</center>

| 方案        | 单卡显存 | 支持 batch size |
|-------------|----------|-----------------|
| 单 GPU      | 8GB      | 2               |
| DDP（4 卡） | 8GB      | 2（每卡）       |
| ZeRO-2（4 卡） | 6GB   | 4（每卡）       |
| ZeRO-3（4 卡） | 4GB   | 8（每卡）       |

(4) 多节点训练
(5) 分布式训练最佳实践
  - Batch Size 调整: 总 `batch size = per_device_batch_size × num_gpus × gradient_accumulation_steps`
  - 学习率缩放: 使用线性缩放规则: `lr_new = lr_base × sqrt(total_batch_size_new / total_batch_size_base)`
  - 监控和调试

#### 11.6.4 生产部署
(1) 模型导出
(2) 推理优化
(3) API 服务

### 11.8 本章小结