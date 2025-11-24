## 什么是 YaRN（Yet another RoPE extensioN）

![Image](https://i.ytimg.com/vi/B07MNiKVIkA/maxresdefault.jpg)

![Image](https://i.ytimg.com/vi/GQPOtyITy54/maxresdefault.jpg)

![Image](https://ar5iv.labs.arxiv.org/html/2309.00071/assets/charts/dynamic.png)

简要来说：

* YaRN 是一种用于 Transformer／大型语言模型（LLM）中 **扩展上下文窗口长度（context window）** 的技术。([arXiv][1])
* 它主要针对使用 Rotary Position Embedding (RoPE) 的模型，因为 RoPE 在原始训练里通常有最大位置 embedding 限制。([Adrian][2])
* YaRN 的核心思路包括：

  * 在推理或微调阶段对位置编码（尤其是 RoPE 部分）做**缩放／插值**（例如将 token 位置 “拉伸”到更大数字）以支持比训练时更长的序列长度。([arXiv][3])
  * 要么通过少量微调数据、有时候带推理时动态缩放（比如 “Dynamic Scaling”）来让模型在更长上下文时仍然保持合理效果。([Continuum Labs Training][4])
* 在论文里，作者指出：通过 YaRN，可以 **显著少量训练／少量额外数据** 就扩展 context 长度，而不像以前那样必须重新从头或大量微调。([arXiv][1])
* 它的一个典型应用：让模型从原本比如 32K tokens 的上下文，扩展到 64K、128K tokens 甚至更多。([GitHub][5])

**简化一句话**：YaRN 让模型“看”得更远——可以一次输入更长的文本，而不是被训练时的最大上下文限制死。

---

## 支持 YaRN 的模型／可以扩展到 ~131,072 tokens 的模型

下面是一些 **已知支持或宣称支持**经 YaRN 扩展的模型，并且可以到 **131,072 tokens（≈128K）** 上下文长度左右／以上。
（注意：虽然模型宣称支持，但实际使用中可能还受限于部署框架、硬件、KV cache 等因素）

| 模型                   | 支持情境                                                                                                                       | 备注                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| yarn‑llama2:7b‑128k  | 基于 Llama 2 7B 版本，支持 128K 上下文（131,072 tokens）([Ollama][6])                                                                  | 适合需要非常长输入的小模型场景                        |
| yarn‑llama2:13b‑128k | 同系列 13B，支持 128K 上下文([Ollama][7])                                                                                           | 参数更多但仍声明为 128K 支持                      |
| Qwen/QwQ‑32B         | 宣称 “Context Length: Full 131,072 tokens” 并且 “For prompts exceeding 8,192 tokens, you must enable YaRN” ([Hugging Face][8]) | 这是一个更大模型（32B）支持 128K 上下文的例子            |
| Qwen3‑32B（社区版本）      | 在模型卡中提到 “Supports a context length of up to 131,072 tokens with YaRN (default 32k)” ([Hugging Face][9])                    | 虽然是社区版本，但说明 YaRN 在该生态内也被用来扩展 Qwen 系列模型 |

---

## 使用 YaRN 时你需要注意的关键设置 / 技术细节

* 在模型的 `config.json` 中，通常看到像这样的配置来开启 YaRN：

  ```json
  "rope_scaling": {
     "factor": 4.0,
     "original_max_position_embeddings": 32768,
     "type": "yarn"
  }
  ```

  这个代表：原始最大位置 embedding 是 32768（32K），通过 rope_scaling 的 factor (4.0) 增加到约 4×32K = 128K。([Hugging Face][10])
* 虽然模型宣称支持 128K 或 131K tokens，但实际部署时还要考虑：

  * KV-cache 消耗（上下文越长，KV cache 占用显存越多）
  * 推理框架是否支持这种扩展（例如是否支持修改 RoPE、是否支持大 batch、是否支持 memory efficient attention）
  * 硬件显存是否能撑住
* 在部署（如你用的 TGI / docker）中，如果想用 YaRN 支持更长上下文，需要确认：模型加载、配置、推理框架都正确开启 “rope_scaling” 或类似参数。
* 尽管 YaRN 技术能扩展，但仍建议实践中 **监控输入 token 数量 + max_total_tokens**，不要盲目把输入拉至最大，因为性能／显存可能下降。

---

## 给你的建议（基于你之前场景：3090 + TGI + Qwen3-1.7B）

* 虽然你的模型是 Qwen3-1.7B（目前看起来原生支持 32 768 tokens），但如果你换用支持 YaRN 的模型（如 Qwen3-32B 或其他已扩展模型），理论上你可以达到 ~128K 上下文。
* 在你当前场景里（24GB 显存 + TGI + 反射 agent），即便模型支持 128K，上下文也要谨慎，因为显存、KV cache 会激增。
* 如果你未来准备用一个 **长上下文专用模型**，可选择 “支持 YaRN 的模型” 如上表所列，然后部署时开启 rope_scaling，并设置 `--max-input-length`, `--max-total-tokens` 赶近 100K 如 100000 tokens，但首先要做好显存测试。

---

如果你愿意，我可以帮你 **查一下“Qwen3‐1.7B”是否已经有 YaRN 扩展版本**、以及 **具体在 TGI 部署中如何配置 YaRN（例如 config.json 改动 + TGI 启动参数）**，你看要不要？

[1]: https://arxiv.org/abs/2309.00071?utm_source=chatgpt.com "YaRN: Efficient Context Window Extension of Large Language Models"
[2]: https://www.adrian.idv.hk/2023-11-09-pqfs23-yarn/?utm_source=chatgpt.com "Peng et al (2023) YaRN: Efficient Context Window Extension of Large ..."
[3]: https://arxiv.org/pdf/2309.00071v1?utm_source=chatgpt.com "YaRN: Efficient Context Window Extension of Large Language Models"
[4]: https://training.continuumlabs.ai/training/the-fine-tuning-process/training-processes/yarn-efficient-context-window-extension-of-large-language-models?utm_source=chatgpt.com "YaRN: Efficient Context Window Extension of Large Language Models"
[5]: https://github.com/AI-Natural-Language-Processing-Lab/yarn-Efficient-Context-Window-Extension-of-Large-Language-Models/blob/master/README.md?utm_source=chatgpt.com "yarn-Efficient-Context-Window-Extension-of-Large-Language-Models/README ..."
[6]: https://ollama.com/library/yarn-llama2%3A7b-128k?utm_source=chatgpt.com "yarn-llama2:7b-128k"
[7]: https://ollama.com/library/yarn-llama2%3A13b-128k-q3_K_M?utm_source=chatgpt.com "yarn-llama2:13b-128k-q3_K_M"
[8]: https://huggingface.co/Qwen/QwQ-32B?utm_source=chatgpt.com "Qwen/QwQ-32B · Hugging Face"
[9]: https://huggingface.co/lmstudio-community/Qwen3-32B-GGUF?utm_source=chatgpt.com "lmstudio-community/Qwen3-32B-GGUF · Hugging Face"
[10]: https://huggingface.co/Qwen/QwQ-32B/discussions/28?utm_source=chatgpt.com "Qwen/QwQ-32B · Is this model native 128K context length, or YaRN extended?"
