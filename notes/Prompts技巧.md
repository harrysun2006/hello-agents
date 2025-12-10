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

