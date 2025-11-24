# 第一部分：智能体与语言模型基础
## [第一章 初识智能体](https://datawhalechina.github.io/hello-agents/#/./chapter1/第一章%20初识智能体)
### 1.1 什么是智能体？
在人工智能领域，智能体被定义为任何能够通过传感器（Sensors）感知其所处环境（Environment），并自主地通过执行器（Actuators）采取行动（Action）以达成特定目标的实体。
- 真正赋予智能体"智能"的，是其自主性（Autonomy）

#### 1.1.3 智能体的类型
基于内部决策架构的分类
- 反射智能体（Simple Reflex Agent）
- 基于模型的反射智能体（Model-Based Reflex Agent）, 世界模型（World Model）
- 基于目标的智能体（Goal-Based Agent）
- 基于效用的智能体（Utility-Based Agent）
- 学习型智能体（Learning Agent）, 强化学习（Reinforcement Learning, RL

基于时间与反应性的分类
- 反应式智能体 (Reactive Agents): 追求速度的反应性（Reactivity）
- 规划式智能体(Deliberative Agents): 追求最优解的规划性（Deliberation）
- 混合式智能体(Hybrid Agents): 规划(Reasoning) + 反应(Acting & Observing) 

基于知识表示的分类
- 符号主义 AI（Symbolic AI）
- 亚符号主义 AI（Sub-symbolic AI）
- 神经符号主义 AI（Neuro-Symbolic AI）

### 1.2 智能体的构成与运行原理
#### 1.2.1 任务环境定义
在人工智能领域，通常使用PEAS 模型来精确描述一个任务环境，即分析其性能度量(Performance)、环境(Environment)、执行器(Actuators)和传感器(Sensors) 。

#### 1.2.2 智能体的运行机制
智能体循环 (Agent Loop)
感知 (Perception)/观察 (Observation) => 思考 (Thought): 规划 (Planning) + 工具选择 (Tool Selection) => 行动 (Action) => 环境 (Environment) 的状态变化 (State Change) => 新的观察 (Observation)

## [第二章 智能体发展史](https://datawhalechina.github.io/hello-agents/#/./chapter2/第二章%20智能体发展史)
### 2.1 基于符号与逻辑的早期智能体
#### 2.1.1 物理符号系统假说
1976年由艾伦·纽厄尔（Allen Newell）和赫伯特·西蒙（Herbert A. Simon）共同提出的物理符号系统假说（PhysicalSymbol SystemHypothesis, PSSH）
- 充分性论断：任何一个物理符号系统，都具备产生通用智能行为的充分手段
- 必要性论断：任何一个能够展现通用智能行为的系统，其本质必然是一个物理符号系统。
PSSH大胆地宣称：智能的本质，就是符号的计算与处理。

#### 2.1.2 专家系统
**知识库与推理机**
专家系统的“智能”主要源于其两大核心组件：知识库和推理机。
- 知识库（Knowledge Base）
- 推理机（Inference Engine）: 正向链（Forward Chaining）, 反向链（Backward Chaining）
应用案例与分析：MYCIN系统

#### 2.1.3 SHRDLU
由特里·威诺格拉德（Terry Winograd）于1968-1970年开发, SHRDLU旨在构建一个能在“积木世界”这一微观环境中，通过自然语言与人类流畅交互的综合性智能体
- 自然语言理解
- 规划与行动
- 记忆与问答

#### 2.1.4 符号主义面临的根本性挑战
- 常识知识与知识获取瓶颈
  - 知识获取瓶颈（Knowledge Acquisition Bottleneck）
  - 常识问题（Common-sense Problem）: Cyc项目
- 框架问题与系统脆弱性
  - 框架问题（Frame Problem）: 智能体执行一个动作后，如何高效判断哪些事物未发生改变是一个逻辑难题
  - 系统脆弱性（Brittleness）

### 2.2 构建基于规则的聊天机器人
#### 2.2.1 ELIZA 的设计思想
ELIZA是由麻省理工学院的计算机科学家约瑟夫·魏泽鲍姆（Joseph Weizenbaum）于1966年发布的一个计算机程序[6]，是早期自然语言处理领域的著名尝试之一

#### 2.2.2 模式匹配与文本替换
ELIZA的算法流程基于模式匹配（Pattern Matching）与文本替换（Text Substitution）
- 关键词识别与排序
- 分解规则
- 重组规则
- 代词转换

#### 2.2.3 核心逻辑的实现
运行[mini ELIZA](code/chapter2/ELIZA.py), 可以发现:
- 缺乏语义理解: 无法理解否定词“not”的作用
- 无上下文记忆
- 规则的扩展性问题

### 2.3 马文·明斯基的心智社会
马文·明斯基（Marvin Minsky）没有继续尝试为单一推理核心添加更多规则，而是在他的《心智社会》（The Society of Mind）[7] 一书中提出了一个革命性的问题："What magical trick makes us intelligent? The trick is that there is no trick. The power of intelligence stems from our vast diversity, not from any single, perfect principle."

#### 2.3.1 对单一整体智能模型的反思
- “理解”是什么？
- “常识”是什么？
- 智能体应该如何构建？

#### 2.3.2 作为协作体的智能
- 简单的智能体被组织起来，形成功能更强大的**机构（Agency）**
- **涌现（Emergence）**是理解心智社会理论的关键

#### 2.3.3 对多智能体系统的理论启发
- 分布式人工智能（Distributed Artificial Intelligence, DAI）
- 多智能体系统（Multi-Agent System, MAS）

- 去中心化控制（Decentralized Control）
- 涌现式计算（Emergent Computation）: 蚁群算法、粒子群优化
- 智能体的社会性（Agent Sociality）: 智能体之间的交互（激活、抑制）; 智能体之间的通信语言（如ACL）、交互协议（如契约网）、协商策略、信任模型乃至组织结构

### 2.4 学习范式的演进与现代智能体
#### 2.4.1 从符号到联结
**联结主义（Connectionism）**
- 知识的分布式表示
- 简单的处理单元
- 通过学习调整权重

#### 2.4.2 基于强化学习的智能体
**强化学习（Reinforcement Learning, RL）** 正是专注于解决序贯决策问题的学习范式。强化学习的框架可以用几个核心要素来描述：
- 智能体（Agent）
- 环境（Environment）
- 状态（State, S）
- 行动（Action, A）
- 奖励（Reward, R）

#### 2.4.3 基于大规模数据的预训练
如何让智能体在开始学习具体任务前，就先具备对世界的广泛理解？这一问题的解决方案，最终在**自然语言处理（Natural Language Processing, NLP）**领域中浮现，其核心便是基于大规模数据的**预训练（Pre-training）**。
预训练与微调（Pre-training, Fine-tuning）范式的提出彻底改变了这一现状。其核心思想分为两步：
- **预训练阶段**：**自监督学习（Self-supervised Learning）** 训练一个超大规模的神经网络模型, 常见的目标是“预测下一个词”。
- **微调阶段**

大型语言模型的诞生与涌现能力
- **上下文学习（In-context Learning）**：无需调整模型权重，仅在输入中提供**几个示例（Few-shot）**甚至**零个示例（Zero-shot）**，模型就能理解并完成新的任务。
- **思维链（Chain-of-Thought）推理**：通过引导模型在回答复杂问题前，先输出一步步的推理过程，可以显著提升其在逻辑、算术和常识推理任务上的准确性。

#### 2.4.4 基于大语言模型的智能体
LLM驱动的智能体通过一个由多个模块协同工作的、持续迭代的闭环流程来完成任务:
- **感知 (Perception)** ：流程始于**感知模块 (Perception Module)**。它通过传感器从**外部环境 (Environment)**接收原始输入，形成**观察 (Observation)**。
- **思考 (Thought)** ：这是智能体的认知核心，对应图中的**规划模块 (Planning Module)**和**大型语言模型 (LLM)** 的协同工作。
  - **规划与分解**：**反思 (Reflection)** 和**自我批判 (Self-criticism)**
  - **推理与决策**：LLM 接收来自规划模块的指令，并与**记忆模块 (Memory)** 交互以整合历史信息, 进行深度推理，最终决策出下一步要执行的具体操作，这通常表现为一个**工具调用 (Tool Call)**。
- **行动 (Action)** ：由**执行模块 (Execution Module)** 负责。从**工具箱 (Tool Use)** 中选择并调用合适的工具（如代码执行器、搜索引擎、API等）来与环境交互或执行任务。
- **观察 (Observation)** 与循环

#### 2.4.5 智能体发展关键节点概览
- **符号主义 (Symbolism)** : 司马贺 (Herbert A. Simon) 、明斯基 (Marvin Minsky) 
- **联结主义 (Connectionism)** : 杰弗里·辛顿 (Geoffrey Hinton), 卷积神经网络、Transformer等模型
- **行为主义 (Behaviorism)** : 早期的TD-Gammon, 与深度学习结合并击败人类顶尖棋手的AlphaGo

## [第三章 大语言模型基础](https://datawhalechina.github.io/hello-agents/#/./chapter3/第三章%20大语言模型基础)
### 3.1 语言模型与 Transformer 架构
#### 3.1.1 从 N-gram 到 RNN
**语言模型 (Language Model, LM)** 是自然语言处理的核心，其根本任务是计算一个词序列（即一个句子）出现的概率。
（1）统计语言模型与N-gram的思想
在深度学习兴起之前，统计方法是语言模型的主流。其核心思想是：一个句子出现的概率，等于该句子中每个词出现的条件概率的连乘。
对于一个由词 \( w_1, w_2, \ldots, w_m \) 构成的句子 \( S \)，其概率 \( P(S) \) 可以表示为：

\[
P(S) = P(w_1, w_2, \ldots, w_m)
     = P(w_1)\cdot P(w_2 \mid w_1)\cdot P(w_3 \mid w_1, w_2)\cdots P(w_m \mid w_1, \ldots, w_{m-1})
\]

这个公式被称为**概率的链式法则（Chain Rule）**。

然而，直接计算这个公式几乎是不可能的，因为像  
\( P(w_m \mid w_1, w_2, \ldots, w_{m-1}) \)  
这样的长条件概率太难从语料库中估计：词序列 \( w_1, w_2, \ldots, w_{m-1} \) 可能从未在训练数据中出现过。

**马尔可夫假设 (Markov Assumption)** 。其核心思想是：我们不必回溯一个词的全部历史，可以近似地认为，一个词的出现概率只与它前面有限的 n−1 个词有关
- **Bigram（当 N = 2 时）**：这是最简单的情况，我们假设一个词的出现只与它前面的一个词有关。因此，链式法则中复杂的条件概率  
  \( P(w_i \mid w_1, \ldots, w_{i-1}) \)  
  就可以被近似为更容易计算的形式：

  \[
  P(w_i \mid w_1, \ldots, w_{i-1}) \approx P(w_i \mid w_{i-1})
  \]

- **Trigram（当 N = 3 时）**：类似地，我们假设一个词的出现只与它前面的两个词有关：

  \[
  P(w_i \mid w_1,\ldots, w_{i-1}) \approx P(w_i \mid w_{i-2}, w_{i-1})
  \]

这些概率可以通过在大型语料库中进行**最大似然估计（Maximum Likelihood Estimation, MLE）**来计算。这个术语听起来很复杂，但其思想非常直观：最可能出现的，就是我们在数据中看到次数最多的。

例如，对于 Bigram 模型，我们想计算在词 \( w_{i-1} \) 出现后，下一个词为 \( w_i \) 的概率 \( P(w_i \mid w_{i-1}) \)。根据最大似然估计，这个概率可以通过简单的计数来估算：

\[
P(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}
\]

这里的 `Count()` 函数就代表“计数”：

- `Count(w_{i-1}, w_i)`：表示词对 \((w_{i-1}, w_i)\) 在语料库中连续出现的总次数。
- `Count(w_{i-1})`：表示单个词 \( w_{i-1} \) 在语料库中出现的总次数。

**N-gram** 模型两个致命缺陷：
- **数据稀疏性 (Sparsity)**: 可以通过平滑 (Smoothing) 技术缓解，但无法根除
- **泛化能力差**: 模型无法理解词与词之间的语义相似性

（2）神经网络语言模型与词嵌入
2003年，Bengio 等人提出的**前馈神经网络语言模型 (Feedforward Neural Network Language Model)** 是这一领域的里程碑。核心思想可以分为两步：
- 构建一个语义空间: **词嵌入 (Word Embedding)** 或词向量
- 学习从上下文到下一个词的映射: **余弦相似度 (Cosine Similarity)**
similarity(\(\vec{a}, \vec{b}\)) = \(\cos(\theta)\) = \[\frac{\vec{a} \cdot \vec{b}}{|\vec{a}|\, |\vec{b}|}\]

`vector('King') - vector('Man') + vector('Woman')` ~= `vector('Queen')`

（3）循环神经网络 (RNN) 与长短时记忆网络 (LSTM)
**循环神经网络 (Recurrent Neural Network, RNN)** 应运而生，其核心思想非常直观：为网络增加“记忆”能力。
标准的 RNN 在实践中存在一个严重的问题：长期依赖问题 (Long-term Dependency Problem) 
- **梯度消失**: 梯度值 -> 0
- **梯度爆炸**: 梯度值 -> ∞

**长短时记忆网络 (Long Short-Term Memory, LSTM)**: 解决长期依赖问题
LSTM 是一种特殊的 RNN，其核心创新在于引入了**细胞状态 (Cell State)** 和一套精密的**门控机制 (Gating Mechanism)**
- **遗忘门 (Forget Gate)**：决定从上一时刻的细胞状态中丢弃哪些信息。
- **输入门 (Input Gate)**：决定将当前输入中的哪些新信息存入细胞状态。
- **输出门 (Output Gate)**：决定根据当前的细胞状态，输出哪些信息到隐藏状态。 

#### 3.1.2 Transformer 架构解析
Transformer在2017 年由谷歌团队提出。它完全抛弃了循环结构，转而完全依赖一种名为**注意力 (Attention)** 的机制来捕捉序列内的依赖关系，从而实现了真正意义上的并行计算。
（1）Encoder-Decoder 整体结构
- **编码器 (Encoder)** ：任务是“理解”输入的整个句子。
- **解码器 (Decoder)** ：任务是“生成”目标句子。

（2）从自注意力到多头注意力
想象一下我们阅读这个句子：“The agent learns because it is intelligent.”。当我们读到加粗的 "it" 时，为了理解它的指代，我们的大脑会不自觉地将更多的注意力放在前面的 "agent" 这个词上。**自注意力 (Self-Attention)** 机制就是对这种现象的数学建模。
- **查询 (Query, Q)**：代表当前词元，它正在主动地“查询”其他词元以获取信息。
- **键 (Key, K)**：代表句子中可被查询的词元“标签”或“索引”。
- **值 (Value, V)**：代表词元本身所携带的“内容”或“信息”。

（3）前馈神经网络
在每个 Encoder 和 Decoder 层中，多头注意力子层之后都跟着一个**逐位置前馈网络(Position-wise Feed-Forward Network, FFN)** 。所有位置共享的是同一组网络权重，这个网络的结构非常简单，由两个线性变换和一个 ReLU 激活函数组成：

\[
\text{FFN}(x) = \max(0,\, xW_1 + b_1)\, W_2 + b_2
\]

其中，\(x\) 是注意力子层的输出。  
\(W_1, b_1, W_2, b_2\) 是可学习的参数。

通常，第一个线性层的输出维度 \(d_{\text{ff}}\) 会远大于输入的维度 \(d_{\text{model}}\)  
（例如：`d_ff = 4 * d_model`），  
经过 ReLU 激活后再通过第二个线性层映射回 \(d_{\text{model}}\) 维度。
这种“先扩大再缩小”的模式，被认为有助于模型学习更丰富的特征表示。

（4）残差连接与层归一化
- **残差连接 (Add)**：该操作将子模块的输入 `x` 直接加到该子模块的输出 `Sublayer(x)` 上。这一结构解决了深度神经网络中的**梯度消失 (Vanishing Gradients)** 问题。
- **层归一化 (Norm)**：该操作对单个样本的所有特征进行归一化，使其均值为0，方差为1。这解决了模型训练过程中的**内部协变量偏移 (Internal Covariate Shift)** 问题。

（5）3.1.2.5 位置编码

#### 3.1.3 Decoder-Only 架构
Transformer的设计哲学是“先理解，再生成”。编码器负责深入理解输入的整个句子，形成一个包含全局信息的上下文记忆，然后解码器基于这份记忆来生成翻译。但 OpenAI 在开发 **GPT (Generative Pre-trained Transformer)** 时，提出了一个更简单的思想：**语言的核心任务，不就是预测下一个最有可能出现的词吗？**
无论是回答问题、写故事还是生成代码，本质上都是在一个已有的文本序列后面，一个词一个词地添加最合理的内容。基于这个思想，GPT 做了一个大胆的简化：**它完全抛弃了编码器，只保留了解码器部分**。 这就是 **Decoder-Only** 架构的由来。
Decoder-Only 架构的工作模式被称为**自回归 (Autoregressive)** 。
解码器是如何保证在预测第 `t` 个词时，不去“偷看”第 `t+1` 个词的答案呢？
**掩码自注意力 (Masked Self-Attention)**

**Decoder-Only 架构的优势**
- 训练目标统一
- 结构简单，易于扩展
- 天然适合生成任务

### 3.2 与大语言模型交互
#### 3.2.1 提示工程
（1）模型采样参数
- **`Temperature`**：温度是控制模型输出 “随机性” 与 “确定性” 的关键参数。
传统的 Softmax 概率分布为：

\[
p_i = \frac{e^{z_i}}{\sum_{j=1}^k e^{z_j}}
\]

引入温度参数 \(T > 0\) 后，Softmax 改写为：

\[
p_i^{(T)} = \frac{e^{z_i / T}}{\sum_{j=1}^k e^{z_j / T}}
\]

  * 低温度（0 ⩽ Temperature < 0.3）时输出更 “精准、确定”。适用场景： 事实性任务：如问答、数据计算、代码生成； 严谨性场景：法律条文解读、技术文档撰写、学术概念解释等场景。
  * 中温度（0.3 ⩽ Temperature < 0.7）：输出 “平衡、自然”。适用场景： 日常对话：如客服交互、聊天机器人； 常规创作：如邮件撰写、产品文案、简单故事创作。
  * 高温度（0.7 ⩽ Temperature < 2）：输出 “创新、发散”。适用场景： 创意性任务：如诗歌创作、科幻故事构思、广告 slogan brainstorm、艺术灵感启发； 发散性思考。
- **`Top-k`**: 其原理是将所有 token 按概率从高到低排序，取排名前 k 个的 token 组成 “候选集”，随后对筛选出的 k 个 token 的概率进行 “归一化”: \[\hat{p}_i = \frac{p_i}{\sum_{j \in \text{候选集}} p_j}\]

- **`Top-p`**: 其原理是将所有 token 按概率从高到低排序，从排序后的第一个 token 开始，逐步累加概率，直到累积和首次达到或超过阈值 p：\[\sum_{i \in S} p(i) \ge p\] 此时累加过程中包含的所有 token 组成 “核集合”，最后对核集合进行归一化。

在文本生成中，当同时设置 Top-p、Top-k 和温度系数时，这些参数会按照分层过滤的方式协同工作，其优先级顺序为：Temperature → Top-k → Top-p。温度调整整体分布的陡峭程度，Top-k 会先保留概率最高的 k 个候选，然后 Top-p 会从 Top-k 的结果中选取累积概率≥p 的最小集合作为最终的候选集。

（2）零样本、单样本与少样本提示
- **零样本提示 (Zero-shot Prompting)** 这指的是我们不给模型任何示例，直接让它根据指令完成任务。
- **单样本提示 (One-shot Prompting)** 我们给模型提供一个完整的示例，向它展示任务的格式和期望的输出风格。
- **少样本提示 (Few-shot Prompting)** 我们提供多个示例，这能让模型更准确地理解任务的细节、边界和细微差别，从而获得更好的性能。

（3）指令调优的影响
**指令调优 (Instruction Tuning)** 是一种微调技术，它使用大量“指令-回答”格式的数据对预训练模型进行进一步的训练。

（4）基础提示技巧
**角色扮演 (Role-playing)** 通过赋予模型一个特定的角色，我们可以引导它的回答风格、语气和知识范围，使其输出更符合特定场景的需求。

（5）思维链
**思维链 (Chain-of-Thought, CoT)** 是一种强大的提示技巧，它通过引导模型“一步一步地思考”，提升了模型在复杂任务上的推理能力。实现 CoT 的关键，是在提示中加入一句简单的引导语，如“请逐步思考”或“Let's think step by step”。
```
一个篮球队在一个赛季的80场比赛中赢了60%。在接下来的赛季中，他们打了15场比赛，赢了12场。两个赛季的总胜率是多少？
```
现在 ChatGPT 5.1 和 llama3.1:8b 默认就会使用CoT, 不需要引导语也能分步骤解决问题。

#### 3.2.2 文本分词
将文本序列转换为数字序列的过程，就叫做**分词 (Tokenization)** 。**分词器 (Tokenizer)** 的作用，就是定义一套规则，将原始文本切分成一个个最小的单元，我们称之为**词元 (Token)** 。

##### 3.2.2.1 为何需要分词
早期的自然语言处理任务可能会采用简单的分词策略：
- 按词分词 (Word-based)
- 按字符分词 (Character-based)
为了兼顾词表大小和语义表达，现代大语言模型普遍采用**子词分词 (Subword Tokenization)** 算法。
agent => agent
Tokenization => Token + ization

##### 3.2.2.2 字节对编码算法解析
**字节对编码 (Byte-Pair Encoding, BPE)** 是最主流的子词分词算法之一，GPT系列模型就采用了这种算法。
初始化 --> 迭代合并 --> 重复
- **WordPiece**：Google BERT 模型采用的算法。它与 BPE 非常相似，但合并词元的标准不是“最高频率”，而是“能最大化提升语料库的语言模型概率”。简单来说，它会优先合并那些能让整个语料库的“通顺度”提升最大的词元对。
- **SentencePiece**：Google 开源的一款分词工具，Llama 系列模型采用了此算法。它最大的特点是，将空格也视作一个普通字符（通常用下划线 _ 表示）。这使得分词和解码过程完全可逆，且不依赖于特定的语言（例如，它不需要知道中文不使用空格分词）。

##### 3.2.2.3 分词器对开发者的意义
- **上下文窗口限制**：模型的上下文窗口（如 8K, 128K）是以 **Token 数量**计算的
- **API 成本**：大多数模型 API 都是按 Token 数量计费的
- **模型表现的异常**：有时模型的奇怪表现根源在于分词

#### 3.2.3 调用开源大语言模型

#### 3.2.4 模型的选择
##### 3.2.4.1 模型选型的关键考量
- 性能与能力：LMSys Chatbot Arena Leaderboard
- 成本
- 速度（延迟）
- 上下文窗口
- 部署方式
- 生态与工具链
- 可微调性与定制化
- 安全性与伦理

##### 3.2.4.2 闭源模型概览
- OpenAI GPT 系列: RLHF
- Google Gemini 系列: Gemini Ultra 是其最强大的模型，适用于高度复杂的任务；Gemini Pro 适用于广泛的任务，提供高性能和效率；Gemini Nano 则针对设备端部署进行了优化。
- Anthropic Claude 系列: Claude 3 系列包括 Claude 3 Opus（最智能、性能最强）、Claude 3 Sonnet（性能与速度兼顾的平衡之选）和 Claude 3 Haiku（最快、最紧凑的模型，适用于近乎实时的交互）。最新的 Claude 4 系列模型，如 Claude 4 Opus，在通用智能、复杂推理和代码生成方面取得了显著进展，进一步提升了处理长上下文和多模态任务的能力。
- 国内主流模型：中国在大语言模型领域涌现出众多具有竞争力的闭源模型，以百度文心一言(ERNIE Bot)、腾讯混元(Hunyuan)、华为盘古(Pangu-α)、科大讯飞星火(SparkDesk)和月之暗面(Moonshot AI)等为代表的国产模型，在中文处理上具备天然优势，并深度赋能本土产业。

##### 3.2.4.3 开源模型概览
- Meta Llama 系列
- Mistral AI 系列: 来自法国的 Mistral AI 以其“小尺寸、高性能”的模型设计而闻名。其最新模型 Mistral Medium 3.1 于2025年8月发布，在代码生成、STEM推理和跨领域问答等任务上准确率与响应速度均有显著提升，基准测试表现优于Claude Sonnet 3.7与Llama 4 Maverick等同级模型。它具备原生多模态能力，可同时处理图像与文字混合输入，并内置“语调适配层”，帮助企业更轻松实现符合品牌调性的输出。
- 国内开源力量：国内厂商和科研机构也在积极拥抱开源，例如阿里巴巴的通义千问 (Qwen) 系列和清华大学与智谱 AI 合作的 ChatGLM 系列，它们提供了强大的中文能力，并围绕自身构建了活跃的社区。

### 3.3 大语言模型的缩放法则与局限性
#### 3.3.1 缩放法则
- 模型参数量和训练数据量之间存在一个最优配比
- 缩放法则最令人惊奇的产物是“能力的涌现”。所谓能力涌现，是指当模型规模达到一定阈值后，会突然展现出在小规模模型中完全不存在或表现不佳的全新能力。例如，**链式思考 (Chain-of-Thought)** 、**指令遵循 (Instruction Following)** 、多步推理、代码生成等能力，都是在模型参数量达到数百亿甚至千亿级别后才显著出现的。

#### 3.3.2 模型幻觉
**模型幻觉（Hallucination）**通常指的是大语言模型生成的内容与客观事实、用户输入或上下文信息相矛盾，或者生成了不存在的事实、实体或事件。
- **事实性幻觉 (Factual Hallucinations)**: 模型生成与现实世界事实不符的信息。
- **忠实性幻觉 (Faithfulness Hallucinations)**: 在文本摘要、翻译等任务中，生成的内容未能忠实地反映源文本的含义。
- **内在幻觉 (Intrinsic Hallucinations)**: 模型生成的内容与输入信息直接矛盾。

多种检测和缓解幻觉的方法：
- 数据层面： 通过高质量数据清洗、引入事实性知识以及强化学习与人类反馈 (RLHF) 等方式，从源头减少幻觉。
- 模型层面： 探索新的模型架构，或让模型能够表达其对生成内容的不确定性。
- 推理与生成层面：
  * 检索增强生成 (Retrieval-Augmented Generation, RAG)： 这是目前缓解幻觉的有效方法之一。RAG 系统通过在生成之前从外部知识库（如文档数据库、网页）中检索相关信息，然后将检索到的信息作为上下文，引导模型生成基于事实的回答。
  * 多步推理与验证： 引导模型进行多步推理，并在每一步进行自我检查或外部验证。
  * 引入外部工具： 允许模型调用外部工具（如搜索引擎、计算器、代码解释器）来获取实时信息或进行精确计算。

### 3.4 本章小结
核心知识点回顾：
- 模型演进与核心架构: **统计语言模型 (N-gram)** ==> **神经网络模型 (RNN, LSTM)** ==> **奠定现代 LLM 基础的 Transformer**
- 与模型的交互方式: 与LLM 交互的两个核心环节：**提示工程 (Prompt Engineering)** 和 **文本分词 (Tokenization)**
- 模型生态与选型：为智能体选择模型时需要权衡的关键因素，闭源: OpenAI GPT、Google Gemini; 开源: Llama、Mistral
- 法则与局限：本章探讨了驱动 LLM 能力提升的缩放法则，也分析了模型幻觉、知识过时等固有局限性以及缓解方法


### TODO: 扩展阅读
- [PyTorch Transformer 英中翻译超详细教程](https://zhuanlan.zhihu.com/p/581334630)
- [DL-Demos](https://github.com/SingleZombie/DL-Demos)
- [Transformer模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680)
- [李宏毅老师讲解的Transformer](https://www.youtube.com/watch?v=ugWDIIOHtPA)

### Prompt Engineering 202502v7 (by Lee Boonstra)
- Temperature, Top-K, Top-P, Token Limit
- Zero-shot, One-shot, Few-shot
- System, contextual and role prompting
- Step-back prompting
- Chain of Thought (CoT)
- Tree of Thoughts (ToT)
- ReAct (Reason & Act)
  - TODO: Example & Practice: Creating a ReAct Agent with LangChain and VertexAI (P38)
- Automatic Prompt Engineering (APE)
- Evaluate APE: BLEU (Bilingual Evaluation Understudy) or ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- Multimodal prompting: 
  * Design with simplicity: Try using verbs that describe the action.
  * Be specific about the output
  * Use Instructions over Constraints: 越来越多的研究表明，在提示中关注积极的指令比过度依赖限制更有效。这种方法符合人类更喜欢积极指令而非禁止事项列表的习惯。
  * Control the max token length
  * Use variables in prompts
  * Experiment with input formats and writing styles
  * For few-shot prompting with classification tasks, mix up the classes (start with 6 for few shot examples)
  * Adapt to model updates
  * Experiment with output formats: JSON or XML
  * JSON Repair: json-repair
  * Working with Schemas: JSON Schema; "time-aware" by including datetime formatted date/timestamp
- CoT Best practices: temperature <== 0
- Document the various prompt attempts

### 其他 Prompt 技巧<a id='prompt-other-tricks'></a>
- 在绝大多数真实任务里，使用 Markdown 或结构化格式（如 **强调**、列表、代码块）通常会显著提升 LLM 的稳定性、准确性和可控性。
- 哪些 Markdown 元素提升最大？

按效果强弱排序：
| Markdown 元素 | 效果等级 | 原因 |
|---|---|---|
| **``` 代码块** | ⭐⭐⭐⭐⭐ | 最强边界框，极大提升代码正确性 |
| **列表（- / 1.）** | ⭐⭐⭐⭐ | 让模型更清晰完成步骤任务 |
| **粗体（**重点**）** | ⭐⭐⭐ | 明确“这是重点” |
| **标题（##）** | ⭐⭐⭐ | 强制模型分段、结构化回答 |
| **表格** | ⭐⭐⭐ | 明确 schema |
| 引用（>） | ⭐⭐ | 弱形式边界框 |
| 斜体（*） | ⭐ | 最弱增强 |

- 没有内建标签，标签如<think>, <analysis>, <code> 都是用户级标签，不会触发特殊行为模式，但会触发“风格模式”，对模型的行为有强影响
- 对非常小的模型（1B–3B），慎用标签，它们可能无效甚至有害

