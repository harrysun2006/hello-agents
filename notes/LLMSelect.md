Q: 
1. 请以表格形式列出从早期的LLM，如BERT, GPT, GPT-2, GPT-3, 一直到后来的主要的LLM，如ChatGPT, Qwen, DeepSeek, Grok, Llama, Gemini, Gemma 等模型的发布日期，model size, tensor type, Number of Parameters, Number of Paramaters (Non-Embedding), Number of Layers, Number of Attention Heads (GQA), Context Length，预训练数据量，主要benchmark分数等信息
2. 给定一个GPU，其能支持的模型根据什么估算？
3. 请推荐RTX 4090 上能跑的hugging face 公开模型

先说结论：

* 不是所有模型的细节都公开，尤其是 ChatGPT / Gemini / Grok 这种闭源家伙，所以有些列会是“未公开/大致范围”。
* 4090（24GB）本地玩 LLM 非常够用：7B–9B 全精度没压力，15B 左右要一点技巧，30B–70B 需要量化 + 部分 CPU offload。

我分三块回答：

---

## 1. 从 BERT → GPT → LLaMA → Qwen / DeepSeek / Grok / Gemini / Gemma 的大表

> ⚠️ 说明：
>
> * “tensor type” 写的是**典型训练/推理精度**（真实实现里都可以再量化）。
> * “主要 benchmark” 统一用比较常见的 **MMLU** 或官方代表指标；早期模型（BERT/GPT-1/2）发布时还没有 MMLU。
> * “≈ / 未公开” 就是论文或官方没给死数据。

### 1.1 早期代表模型

| 模型                  | 机构     | 发布时间                   | Model size / 参数量 | Tensor type（典型）              | Non-Embedding 参数 | 层数  | 注意力头数 (GQA) | 上下文长度       | 预训练数据量                                                  | 代表 benchmark                                     |
| ------------------- | ------ | ---------------------- | ---------------- | ---------------------------- | ---------------- | --- | ----------- | ----------- | ------------------------------------------------------- | ------------------------------------------------ |
| **BERT-base**       | Google | 2018-10                | ~110M            | 训练 fp32，推理常用 fp16            | 未单独给出            | 12  | 12（非 GQA）   | 512 tokens  | ~3.3B 词（BooksCorpus + 英文 Wikipedia）                     | 当年在 GLUE 等上 SoTA（未有 MMLU）                        |
| **GPT-1 (117M)**    | OpenAI | 2018-06                | 117M             | 训练 fp32，推理常用 fp16            | 未公开              | 12  | 12          | 512 tokens  | BooksCorpus + Wikipedia 等（几十 GB 级）                      | 主要在多任务语言理解上提升，没 MMLU 数据                          |
| **GPT-2-XL (1.5B)** | OpenAI | 2019-02 / 2019-11 完整开源 | 1.5B             | fp32 训练，推理多用 fp16            | 未公开              | 48  | 25          | 1024 tokens | WebText (~8M 网页，约 40GB 文本)                              | 主要看 zero-shot perplexity/语言建模指标，没 MMLU           |
| **GPT-3 (175B)**    | OpenAI | 2020-05                | 175B             | 训练混合精度（fp16/bf16），推理 fp16 为主 | 未公开              | ~96 | ~96         | 2048 tokens | ~300B tokens（CommonCrawl, WebText2, Books, Wikipedia 等） | 论文中在多任务上 SoTA；MMLU 出现后大约是中等水平（具体数多为三方测评，官方未系统给出） |

### 1.2 近几年的主流家族

| 模型                 | 机构       | 发布时间               | 参数量 / Model size                      | Tensor type（典型）              | Non-Embedding 参数                   | 层数                      | 注意力头（GQA）                                            | 上下文长度                                                              | 预训练数据量                             | 主要 benchmark（典型）                                      |
| ------------------ | -------- | ------------------ | ------------------------------------- | ---------------------------- | ---------------------------------- | ----------------------- | ---------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------- | ----------------------------------------------------- |
| **ChatGPT（产品）**    | OpenAI   | 2022-11-30 首次公开    | 基于 GPT-3.5 → GPT-4 系列 → 现为 GPT-5.1 引擎 | 内部训练 bf16/混合精度，服务侧多用 fp16/量化 | 未公开                                | 未公开                     | 未公开                                                  | 早期 ~4k / 8k，现 GPT-4.1 / GPT-5 系列支持最高 10^5–10^6 级别上下文（API / 内部配置不同） | 未公开                                | 官方公布大量综合基准，整体处于闭源 SOTA 阶段；具体架构/参数未开源                  |
| **Llama-3 8B**     | Meta     | 2024-04-18         | 8B                                    | bf16/fp16                    | 未公开                                | ~32                     | 多头注意力，使用 GQA（具体头数未在官方简表中详细列出）                        | 8k tokens（有扩展版本到更长）                                                | ~15T tokens（Llama-3 家族）            | MMLU ≈ 67–68%（macro / micro 平均）                       |
| **Qwen2.5-7B**     | 阿里       | 2024-09-19 （7B 基座） | **7.61B**                             | bf16/fp16                    | **6.53B** non-embedding            | **28**                  | **GQA：Q 28 头 / KV 4 头**                              | **131,072 tokens**（base 支持 131k，Instruct 常用 8k–32k）                | 家族总计 ~18T tokens（Qwen2.5 技术报告）     | MMLU：官方报告约 **74.2%**（7B 基座）                           |
| **DeepSeek-V2**    | DeepSeek | 2024-05-06         | **236B 总参数 / 每 token 激活 21B（MoE）**    | 训练 bf16，推理常配合 fp16/INT8      | 非 MoE 结构下难直接定义；论文未给清晰非 embedding 数 | MoE 结构，等效约 60+ 层（按专家路由） | 使用 Multi-head Latent Attention（MLA），KV 压缩，不能简单写“多少头” | 128k tokens                                                        | **8.1T tokens**                    | 官方：MMLU 与当时 SOTA 开源模型相当，在中文 C-Eval / CMMLU 上领先        |
| **Grok-1**         | xAI      | 2023-11 公布         | ~314B MoE，总参数（非正式）                    | 训练 bf16，推理 fp16              | 未公开                                | 未公开                     | MoE + 自定义路由                                          | 官方提到 8k–32k 量级（具体值未标准化公开）                                          | 未公开（据称 > 数万亿 tokens）               | **MMLU ≈ 73%**，HumanEval ≈ 63.2%                      |
| **Gemini 1.5 Pro** | Google   | 2024-02-15 公布      | 参数量未公开（MoE，大约百 B 级）                   | 训练 bf16/混合精度                 | 未公开                                | 未公开                     | MoE + 多模态注意力                                         | 上下文 **最高可达 1M tokens（企业版）**，常规 128k–1M 范围                          | 训练数据规模未严格公开，量级为数十万亿 tokens（文本+多模态） | MMLU：官方 5-shot ~ **86%** 左右                           |
| **Gemma 2 9B**     | Google   | 2024-06-27         | **9B**                                | bf16/fp16                    | 未公开                                | **42 层**                | GQA：32 attention heads / 8 KV heads                  | 8,192 tokens                                                       | **8T tokens**（9B 版本）               | 技术报告：综合 Elo ≈ 1187，接近 GPT-4-0314，在同尺寸段显著优于 Llama-3-8B |

> ⚠️ 很多闭源模型（ChatGPT 当前内核、Grok、Gemini）的 **精确参数量 / 层数 / non-embedding 参数** 官方都没给，你在做 capacity 估算时只能把它们当作“黑盒 API”，而不是本地可跑模型。

---

## 2. 一块 GPU 能跑多大的模型？怎么估算？

粗略的思路就是看 **显存 = 权重 + KV cache + 激活/中间状态 + 额外开销**。

### 2.1 权重占用（最简单的估算）

对于一个有 (P) 个参数的 dense 模型：

[
\text{Weights VRAM} \approx P \times \text{bytes per param}
]

* fp16 / bf16：2 bytes / param
* fp32：4 bytes / param
* INT8：1 byte / param
* INT4：0.5 byte / param

粗略估算（只看权重）：

* **7B 模型**

  * fp16：7B × 2B ≈ 14 GB
  * INT8：≈ 7 GB
  * INT4：≈ 3.5 GB
* **13B 模型**

  * fp16：≈ 26 GB → 已经超过 4090 单卡 24GB，只能：

    * 用 8bit / 4bit
    * 或者部分权重 / KV cache offload 到 CPU
* **30B 模型**

  * fp16：≈ 60 GB（显然不行）
  * INT4：≈ 15 GB（只算权重的话是可以塞进 24GB 的）

实际运行还要给 **KV cache + activations + 框架开销** 留出 20–50% 空间，所以常见经验是：

> **估算显存 ≈ 1.3–1.8 × 权重占用**

很多实测/经验文章也类似结论：4090 做本地推理，**7B–9B fp16** 很舒服，**30B INT4** 或更大模型需要 carefully 配置和 offload。

### 2.2 KV cache / 上下文长度的影响

KV cache 大小和：

* 层数 L
* 隐藏维度 (d)
* 上下文长度 T
* 精度（fp16/int8…)
* batch size

成正比。非常粗的公式（只为直觉）：

[
\text{KV cache} \approx 2 \times L \times T \times d \times \text{bytes_per_elem}
]

例如：Llama-3-8B（约 32 层）、hidden dim ~4096，fp16，T = 8k：

* KV cache 大约就需要 **几 GB** 的量级。
* 如果你把 context 从 4k 提到 16k，KV cache 理论上**翻 4 倍**，显存压力直线上去。

一些第三方测试表明，Llama-3-8B-Instruct 在 8k 上下文 fp16 推理，大概需要 ~16GB VRAM，和上面估算是一致的。

类似，Gemma-2-9B 在 8k ctx 的 fp16 推理需要 ~23GB VRAM，这也是为什么它刚好塞满 24GB 卡。

### 2.3 MoE 模型怎么估算？

像 DeepSeek-V2、Grok-1 这种 **MoE** 模型：

* 总参数 200B+，但 **每个 token 只激活 20B 左右**。
* 显存主要看“**激活参数量 × 精度**”，而不是总参数。

如果你拿到的是官方提供的 **“激活参数 × bits”**，就可以按 dense 模型类似的方法算。DeepSeek-V2-Lite 的 16B 模型就特意设计成可以在 40GB 卡上跑。

---

## 3. RTX 4090（24GB）上推荐可跑的 Hugging Face 模型

假设你的目标是：

* 本地推理 + 轻度微调（LoRA / PEFT）
* 使用 `transformers` / vLLM / llama.cpp / LMStudio / Ollama 之类的工具链
* 4090 有 24GB 显存

### 3.1 “原生 Transformers / fp16”为主的组合

**1）通用英文/多语对话**

* `meta-llama/Meta-Llama-3-8B-Instruct`

  * 优点：MMLU ~68%，综合能力强，生态非常成熟。
  * 4090 上 fp16 8k ctx + batch size 1–2 基本没压力。

* `google/gemma-2-9b-it`

  * 能力接近甚至略超 Llama-3-8B，在推理 / coding 等任务上有优势。
  * 8k ctx fp16 约 23GB VRAM，用得比较满，但 4090 仍可跑。

* `microsoft/Phi-3-medium-4k-instruct`（14B）

  * 14B dense 模型，4090 上一般建议：

    * 用 **8bit 或 4bit 量化**（bitsandbytes / GPTQ / GGUF），
    * 或者使用像 NVIDIA NGC 上的 int4 RTX 优化版。
  * 优点：推理质量非常高，特别是 reasoning 类任务。

* `mistralai/Mistral-7B-Instruct-v0.3`

  * 7B，性能稳，生态工具非常多。

**2）中文 / 中英文混合对话**

* `Qwen/Qwen2.5-7B-Instruct`

  * 7.61B 参数，131k 上下文支持，中文 + 英文都很好。
  * MMLU ≈ 74% 左右，在同尺寸模型里非常强。
  * 在 4090 上 fp16 完全没问题，长 context 时注意 KV cache。

* `deepseek-ai/DeepSeek-V2-Lite` （或各类 distill 版本）

  * MoE 架构，16B 总参数，激活参数少，专门设计成单 40GB 卡可跑；在 4090 上用 8bit / 4bit 会比较舒服。
  * 适合中文为主、兼顾英文和代码的场景。

### 3.2 量化（GGUF / GPTQ / EXL2）后可尝试的大模型

如果你愿意用 **llama.cpp / LM Studio / Ollama** 这类 runner，结合 **INT4 / INT8 量化**，4090 可以玩到：

* Llama-3-70B 系列的高质量 Q4_K_M / Q6_K 量化版本
* DeepSeek-V2 / V3 的部分量化权重
* Qwen2.5-72B 等

社区和硬件推荐文章一般会说：4090 可以“**轻松跑 30B quant，甚至尝试 70B quant**”。

使用量化时的实践建议：

* 模型文件大小大致 ≈ 所需显存（GGUF）：

  * 比如 Gemma-2-9B Q4 GGUF 文件 ~7GB，那么 24GB VRAM 足够支撑长上下文 + 较大 batch。
* 记得留出 **2–4GB** 给系统、loader、KV cache buffer，模型文件不要刚好等于显存大小。

### 3.3 粗略“配机规则”总结（以 24GB 4090 为例）

* **完全 fp16 / bf16：**

  * 7B：非常轻松，长上下文也行
  * 8–9B：可行，但 8k 以上上下文要注意 batch size
  * 14B：基本要用 8bit 或部分 offload
* **INT8：**

  * 14B：可玩
  * 30B：在合理上下文和 batch 下也能工作
* **INT4：**

  * 30B：比较稳
  * 70B：可以跑，但需要：

    * 小 batch
    * 合理上下文（比如 ≤8k）
    * 或者部分权重 / KV cache 放到 CPU

---

如果你愿意，我可以根据你**具体的显存占用情况（nvidia-smi 截图）+ 目标任务（中文对话 / 代码 / 长文理解）**，帮你配置一套更精准的：

* 模型选择（具体 HF repo 名）
* `transformers` / vLLM / llama.cpp 的加载参数（dtype、device_map、max_seq_len）
* 以及如何检测/调试显存 OOM 的步骤。
