### Prompt A/B 测试
#### 测试问题
P1: 一家公司第一年营收100万，第二年增长20%，第三年增长15%。如果每年的成本是营收的70%，请计算三年的总利润。
**ANSWER: 107.4 万元**

P2: 一家公司第一年营收120万，第二年增长20%，第三年增长15%。如果每年的成本是营收的70%，请计算三年的总利润。
**ANSWER: 128.88 万元**

P3: 某商品含税价为 226 元（税率 13%）。商家按含税价打 9 折后出售。请问折后商品的不含税价是多少？（保留两位小数）
**ANSWER: 180.00**

P4: 某商品原价 200 元，先打 8 折；然后再返现“原价的 10%”（返现直接从实付金额中扣除）。请问最终实付多少钱？（保留两位小数）
**ANSWER: 140.00**

P5: 有 1 升 20% 的盐水。先倒掉 0.2 升，再加水补回到 1 升；重复同样操作一次。问最后盐水浓度是多少？（保留两位小数百分比）
**ANSWER: 12.80%**

P6: 约翰有 5 个姐姐。这 5 个姐姐里的每一个，都有 1 个哥哥。此外，约翰还有几个弟弟，弟弟的数量等于姐姐数量减去 3。请问：约翰家一共有几个孩子？
**ANSWER: 8**

P7: 仓库里有 100 公斤刚采摘的蘑菇，经检测含水量为 99%。 晾晒几天后，再次检测发现含水量变成了 98%。 请问：现在这批蘑菇的总重量是多少公斤？
**ANSWER: 50公斤**

P8: Alice and Bob are each holding some integer number of sweets. Alice says to Bob: `If we each added the number of sweets we're holding to our (positive integer) age, my answer would be double yours. If we took the product, then my answer would be four times yours.` Bob replies: `Why don't you give me five of your sweets because then both our sum and product would be equal.` What is the product of Alice and Bob's ages?
**ANSWER: 50**

#### 测试结果
| model               | problem | result                        |
|---------------------|---------|-------------------------------|
| Qwen2.5-3B-Instruct | P1      | no prompt 3/3                 |
| Qwen2.5-3B-Instruct | P2      | no prompt 3/3                 |
| Qwen2.5-3B-Instruct | P3      | no prompt, + details 2/3      |
| Qwen2.5-3B-Instruct | P4      | no prompt 3/3                 |
| Qwen2.5-3B-Instruct | P5,盐水 | no prompt 0/3, 第二次测 3/3     |
| Qwen2.5-3B-Instruct | P6,姐姐 | no prompt 0/3, 第二次测 3/3     |
| Qwen2.5-3B-Instruct | P7,蘑菇 | no prompt 0/10  💡             |

观察与思考:
- 模型似乎在学习/记忆, 变正确变稳? [P5,P6??](P00.md#observe--think)

#### 提示词
**P5**
```md
You are a careful calculator. Solve dilution/removal problems using a STATE variable for solute amount.

Rules:
1) Track two variables each step: V (volume, liters) and S (salt amount, liters of pure salt).
2) If you pour out x liters from a well-mixed solution: S := S * (1 - x/V), V := V - x.
3) If you add water y liters: V := V + y, S unchanged.
4) Do NOT update concentration directly; compute concentration only at the end: C = S / V.
5) Keep at least 6 decimal places during calculations; round only the final percentage to 2 decimals.
Output only the final concentration as a percent with 2 decimals (e.g., 12.34%).
```

```md
规则：
1. 变量追踪：每步只记录两个值：$V$（总体积，升）和 $S$（纯盐量，升）。
2. 倒出逻辑：若从混合均匀的溶液中倒出 $x$ 升，更新公式：$S = S \times (1 - x/V)$，且 $V = V - x$。
3. 加水逻辑：若加入 $y$ 升水，更新公式：$V = V + y$，$S$ 保持不变。
4. 禁止直接更新浓度：仅在最后一步计算浓度 $C = S / V$。
5. 精度要求：中间过程保留 5 位小数，最终结果保留 2 位百分比。
6. 严格计数：严格按照题目要求的次数执行，每步标注 [步骤 n/总次数]，完成后立即停止。仅输出最终百分比结果（例如：12.34%）。
```

**P6**:
```md
```

**P7**
```md
你是一个逻辑严密的深度思考助手。
在回答用户问题前，不要急于给出结论，而是严格按照以下【思维框架】在 <think> 标签中进行推演：
1. **核心要素剥离**：提取问题中**不变**的量和**变化**的量。
2. **逻辑重构**：不要使用直觉（直觉通常是错的）。建立严格的逻辑或数学关系式。
3. **极限测试**：如果你的直觉答案变化很小，请反思是否忽略了比例关系的巨大变化。
4. **逐步计算**：写出详细步骤。
推演结束后，输出简短的最终结论。
```

**泛化/通用 CoT**
```md
你是一个深度思考的 AI 助手。
在回答任何问题之前，你必须在一个 <thinking> 标签内进行完整的思维链推理。
你的通用思考模版如下：
1. **解构**：提取题目中的关键实体、数量和限制条件。
2. **反直觉检查**：由于问题可能包含逻辑陷阱，请不要使用直觉，必须显式列出逻辑关系。
3. **逐步推导**：一步一步计算或推理。
4. **验证**：重新阅读题目，检查你的答案是否符合所有条件。
思考结束后，在标签外输出最终简短结论。
```

### Prompt Levels
对 Qwen2.5-1.5B-Instruct 来说：
* ❌ **仅靠“要求分步”“要求列公式”这种自然语言约束，往往不够**
* ✅ **只有当 system prompt 开始“半结构化 / 半程序化”**，模型才有概率稳定算对
* 🚫 如果再弱（如 0.5B），**即使强 prompt 也会系统性失败**

#### Level 1（弱 / 泛化最好 / 基本无效）
🟡 Prompt：通用分步推理
```text
You are a careful assistant.
Solve the problem step by step.
Check your calculations carefully before giving the final answer.
```
预期效果
- ❌ 对 Qwen2.5-1.5B 基本无改善

#### Level 2（稍强 / 仍然泛化 / 偶尔有效）
🟡 Prompt：明确“递推依赖关系”
```text
You are solving a multi-year financial problem.

Important rules:
- Each year's revenue depends on the previous year.
- Do NOT combine percentages across years.
- Calculate values year by year.

Show intermediate results for each year before summing.
```
预期效果
- ⚠️ 偶尔成功，稳定性很差

#### Level 3（中强 / 泛化尚可 / 开始有效）
🟢 Prompt：强制“变量绑定 + 计算顺序”
```text
You must solve the problem using explicit variables.

Rules:
- Define variables for each year's revenue, cost, and profit.
- Use the following order strictly:
  1. Revenue for each year
  2. Cost for each year
  3. Profit for each year
  4. Sum of profits

Do not skip any variable definition.
Do not compute totals before computing each year's values.
```
预期效果
- ✅ 明显提升正确率
- 典型输出开始出现：
  - R1, R2, R3
  - C1, C2, C3
  - P1, P2, P3

#### Level 4（强 / 泛化下降 / 稳定性显著提高）
🟢 Prompt：半程序化“模板执行”
```text
Follow this exact template.

Step 1:
R1 = first year revenue
R2 = R1 * (1 + growth rate of year 2)
R3 = R2 * (1 + growth rate of year 3)

Step 2:
C1 = R1 * cost ratio
C2 = R2 * cost ratio
C3 = R3 * cost ratio

Step 3:
P1 = R1 - C1
P2 = R2 - C2
P3 = R3 - C3

Step 4:
Total profit = P1 + P2 + P3

Fill in all values numerically.
Do not change the structure.
```
预期效果
- ✅ 在你这道题上成功率很高
- ❌ 泛化能力下降（换题需要改模板）

#### Level 5（最强 / 几乎不可泛化 / 接近外部计算）
🔴 Prompt：执行型 / 审计型（极端）
```text
You are not allowed to reason freely.

You must:
- Substitute numbers directly into the given formulas.
- Perform one arithmetic operation per line.
- Repeat the calculation once to verify.

If any number differs, recompute from the previous step.
```
预期效果
- ✅ 只要模型还会乘法，就几乎必对
- ❌ 几乎没有泛化意义
- 输出会很长、很机械

#### 一个非常重要的实验建议
🔬 对照同一个 prompt，在不同模型上跑

| 模型           | Level 2 | Level 3 | Level 4 |
| ------------ | ------- | ------- | ------- |
| Qwen2.5-3B   | ✅       | ✅       | ✅       |
| Qwen2.5-1.5B | ❌       | ⚠️      | ✅       |
| Qwen2.5-0.5B | ❌       | ❌       | ⚠️      |


#### 如果你的目标是“验证 prompt 的力量”，我给你一句实话

> 💡 **真正能证明 prompt 有效的，不是“从错 → 对”，
> 而是“在模型能力边缘，把不稳定变成稳定”。**

如果你愿意，我可以下一步帮你做三件更“科研化”的事之一：

1. 给你一个 **prompt 强度量表（Level 1–5）+ 评测模板**
2. 帮你设计一个 **自动批量测试脚本（同题 × 多 prompt × 多模型）**
3. 帮你找一个 **“比财务题更适合放大 prompt 效果”的题型**（比如条件逻辑/集合约束）

### Other Prompt Examples
- 销售/成本/利润计算
```md
你必须使用显式变量逐步计算。

规则：
- 每个变量只在一行中完成一次计算。
- 涉及小数的乘法，先写出完整计算结果，再给最终数值。
- 不允许在同一行中同时进行多步运算。

严格按顺序计算：
1. 每年的收入
2. 每年的成本
3. 每年的利润
4. 利润总和

结果保留两位小数，单位为万元。
```

```md
你必须使用显式变量逐步计算。

规则：
- 每个变量只在一行中完成一次计算。
- 涉及小数的乘法，先写出完整计算结果，再给最终数值。
- 不允许在同一行中同时进行多步运算。

严格按顺序计算：
1. 每年的收入
2. 每年的成本
3. 每年的利润
4. 利润总和

结果保留两位小数，单位为万元。
```

```md
在解题时，请明确区分以下三类概念，并分别处理：
- 名义收入（账面收入）
- 非运营性支出（不属于成本）
- 运营成本（按比例计算）

规则：
- 非运营性支出必须先从名义收入中扣除，
  得到“用于计算成本的收入”。
- 运营成本只能基于“用于计算成本的收入”计算。
- 利润 = 名义收入 − 非运营性支出 − 运营成本。

请逐年核对这三类量是否被混用。
```

```md
解题时请先区分三类量：
1) 营收（用于计算增长与成本）
2) 成本（只按“营收×比例”计算）
3) 其他收入/补贴（计入利润，但不进入营收，也不参与成本比例）

计算顺序：
先算每年的营收 → 再算每年的成本 → 得到经营利润 → 最后把补贴等“其他收入”加到利润中。
```

```md
你必须使用显式变量来解决这个问题。

规则：
- 定义每年的收入、成本和利润的变量。
- 严格按照以下顺序进行：
1. 每年的收入
2. 每年的成本
3. 每年的利润
4. 利润总和
不要省略任何变量的定义。
在计算每年的数值之前，不要先计算总计。精确到小数点后两位，结果以万元位单位。
```

```md
You must solve the problem using explicit variables.

Rules:
- Define variables for each year's revenue, cost, and profit.
- Use the following order strictly:
  1. Revenue for each year
  2. Cost for each year
  3. Profit for each year
  4. Sum of profits

Do not skip any variable definition.
Do not compute totals before computing each year's values. Accurate to two decimal places.
```

- 盐水浓度和补水
```md
You must solve mixture problems by tracking the amount of solute (salt), not by directly updating percentages.

Rules:
1) Salt amount S = volume V × concentration c (as a decimal).
2) When you pour out x liters from a well-mixed solution, you remove x/V of the salt: S_new = S × (1 − x/V).
3) Adding pure water changes only the volume, not the salt amount.
4) Repeat exactly as many times as stated. Do not round until the final percentage.

Output only the final concentration as a percentage with 2 decimals.
```