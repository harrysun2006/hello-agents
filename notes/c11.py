### chapter 11

import json, re
from datasets import Dataset
from hello_agents.rl import format_math_dataset
from hello_agents.tools import RLTrainingTool
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from pathlib import Path

## 11.2.3 自定义数据集和奖励函数

rl_tool = RLTrainingTool()
sft_dataset = None
rl_dataset = None

# (1) 使用 format_math_dataset 转换
def t01():
    global sft_dataset, rl_dataset
    # 1. 准备原始数据
    custom_data = [
        {
            "question": "What is 2+2?",
            "answer": "2+2=4. #### 4"
        },
        {
            "question": "What is 5*3?",
            "answer": "5*3=15. #### 15"
        },
        {
            "question": "What is 10+7?",
            "answer": "10+7=17. #### 17"
        }
    ]

    # 2. 转换为Dataset对象
    raw_dataset = Dataset.from_list(custom_data)

    # 3. 转换为SFT格式
    sft_dataset = format_math_dataset(
        dataset=raw_dataset,
        format_type="sft",
        model_name="Qwen/Qwen3-0.6B"
    )
    print(f"SFT数据集: {len(sft_dataset)}个样本")
    print(f"字段: {sft_dataset.column_names}")

    # 4. 转换为RL格式
    rl_dataset = format_math_dataset(
        dataset=raw_dataset,
        format_type="rl",
        model_name="Qwen/Qwen3-0.6B"
    )
    print(f"RL数据集: {len(rl_dataset)}个样本")
    print(f"字段: {rl_dataset.column_names}")

# (2) 直接传入自定义数据集
def t02():

    # SFT训练
    result = rl_tool.run({
        "action": "train",
        "algorithm": "sft",
        "model_name": "Qwen/Qwen3-0.6B",
        "output_dir": "./models/custom_sft",
        "num_epochs": 3,
        "batch_size": 4,
        "use_lora": True,
        "custom_dataset": sft_dataset  # 直接传入自定义数据集
    })
    print(result)

    # GRPO训练
    result = rl_tool.run({
        "action": "train",
        "algorithm": "grpo",
        "model_name": "Qwen/Qwen3-0.6B",
        "output_dir": "./models/custom_grpo",
        "num_epochs": 2,
        "batch_size": 2,
        "use_lora": True,
        "custom_dataset": rl_dataset  # 直接传入自定义数据集
    })
    print(result)

# (3) 注册自定义数据集(推荐)
def t03():
    # 1. 注册数据集
    rl_tool.register_dataset("my_math_dataset", rl_dataset)

    # 2. 使用注册的数据集
    result = rl_tool.run({
        "action": "train",
        "algorithm": "grpo",
        "model_name": "Qwen/Qwen3-0.6B",
        "dataset": "my_math_dataset",  # 使用注册的数据集名称
        "output_dir": "./models/custom_grpo",
        "num_epochs": 2,
        "use_lora": True
    })
    print(result)

# 自定义数据集和奖励函数示例
def t05():
    # 1. 准备自定义数据
    custom_data = [
        {"question": "What is 2+2?", "answer": "2+2=4. #### 4"},
        {"question": "What is 5+3?", "answer": "5+3=8. #### 8"},
        {"question": "What is 10+7?", "answer": "10+7=17. #### 17"}
    ]

    # 2. 转换为训练格式
    raw_dataset = Dataset.from_list(custom_data)
    rl_dataset = format_math_dataset(raw_dataset, format_type="rl")

    # 3. 定义自定义奖励函数
    def tolerant_reward(completions: List[str], **kwargs) -> List[float]:
        """带容差的奖励函数"""
        ground_truths = kwargs.get("ground_truth", [])
        rewards = []

        for completion, truth in zip(completions, ground_truths):
            numbers = re.findall(r'-?\d+\.?\d*', completion)
            if numbers:
                try:
                    pred = float(numbers[-1])
                    truth_num = float(truth)
                    error = abs(pred - truth_num)

                    if error < 0.01:
                        reward = 1.0
                    elif error < 5.0:
                        reward = 0.5
                    else:
                        reward = 0.0
                except ValueError:
                    reward = 0.0
            else:
                reward = 0.0

            rewards.append(reward)

        return rewards

    # 4. 创建工具并注册
    rl_tool = RLTrainingTool()
    rl_tool.register_dataset("my_dataset", rl_dataset)
    # 奖励函数会自动使用与dataset同名的注册函数
    # 如果奖励函数与数据集同名, rl_tool.run 时可以只指定 dataset
    rl_tool.register_reward_function("my_dataset", tolerant_reward)
    # rl_tool.run 并不支持 reward 参数
    # rl_tool.register_reward_function("my_reward", tolerant_reward)

    # 5. 训练
    result = rl_tool.run({
        "action": "train",
        "algorithm": "grpo",
        "model_name": "Qwen/Qwen3-0.6B",
        "dataset": "my_dataset",
        # "reward": "my_reward",
        "output_dir": "./models/custom_grpo",
        "num_epochs": 2,
        "batch_size": 2,
        "use_lora": True
    })
    print(result)

## 11.3.1 为什么需要 SFT

# SFT训练
def t10():
    # 创建训练工具
    rl_tool = RLTrainingTool()

    # SFT训练
    result = rl_tool.run({
        # 训练配置
        "action": "train",
        "algorithm": "sft",
        
        # 模型配置
        "model_name": "Qwen/Qwen3-0.6B",
        "output_dir": "./models/sft_model",
        
        # 数据配置
        "max_samples": 100,     # 使用100个样本快速测试
        
        # 训练参数
        "num_epochs": 3,        # 训练3轮
        "batch_size": 4,        # 批次大小
        "learning_rate": 5e-5,  # 学习率
        
        # LoRA配置
        "use_lora": True,       # 使用LoRA
        "lora_rank": 8,         # LoRA秩
        "lora_alpha": 16,       # LoRA alpha
    })

    print(f"\n✓✓✓ 训练完成!")
    print(result)
    # print(f"  - 模型保存路径: {result['model_path']}")
    # print(f"  - 训练样本数: {result['num_samples']}")
    # print(f"  - 训练轮数: {result['num_epochs']}")
    # print(f"  - 最终损失: {result['final_loss']:.4f}")

# 直接用预训练模型解决 GSM8K 问题
def t11():
    # 加载预训练模型
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 测试问题
    question = """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""

    # 构造输入
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    # 生成回答
    outputs = model.generate(**inputs, max_new_tokens=500)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("\n✅ 预训练模型的回答:")
    print(response)

# 直接用预训练模型 或者 SFT 后的模型解决 GSM8K 问题
def t12(name: str = None):
    orig_model = "Qwen/Qwen3-0.6B"
    if name is None: 
        model_name = orig_model
    else:         
        # 加载SFT模型(假设已经训练好)
        model_name = f"./models/{name}"
        if not Path(model_name).exists(): model_name = orig_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 测试问题
    question = """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""

    # 构造输入
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    # 生成回答
    outputs = model.generate(**inputs, max_new_tokens=500)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print(f"\n✅ 模型{model_name}的回答:")
    print(response)

## 11.3.3 SFT 训练实战
# 完整的 SFT 训练，使用全部数据和最佳实践:
def t15():
    rl_tool = RLTrainingTool()

    # 完整SFT训练
    result = rl_tool.run({
        "action": "train",
        "algorithm": "sft",

        # 模型配置
        "model_name": "Qwen/Qwen3-0.6B",
        "output_dir": "./models/sft_full2",

        # 数据配置
        "max_samples": None,    # 使用全部数据(7473个样本)

        # 训练参数
        "num_epochs": 3,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,

        # LoRA配置
        "use_lora": True,
        "lora_rank": 16,        # 使用更大的rank
        "lora_alpha": 32,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

        # 其他配置
        "save_steps": 500,      # 每500步保存一次
        "logging_steps": 100,   # 每100步记录一次
        "eval_steps": 500,      # 每500步评估一次
    })
    result_dict = json.loads(result)
    print(f"训练完成: {result_dict}")

# 使用 HelloAgents 评估模型
def t21():
    rl_tool = RLTrainingTool()

    # 评估SFT模型
    eval_result = rl_tool.run({
        "action": "evaluate",
        "model_path": "./models/sft_full",
        "max_samples": 100,     # 在100个测试样本上评估
        "use_lora": True,
    })

    eval_data = json.loads(eval_result)
    print(f"\n评估结果:")
    print(f"  - 准确率: {eval_data['accuracy']}")
    print(f"  - 平均奖励: {eval_data['average_reward']}")
    print(f"  - 测试样本数: {eval_data['num_samples']}")

# 对比不同阶段的模型, 理解 SFT 的效果
def t22():
    # 评估预训练模型(未经SFT)
    base_result = rl_tool.run({
        "action": "evaluate",
        "model_path": "Qwen/Qwen3-0.6B",
        "max_samples": 100,
        "use_lora": False,
    })
    base_data = json.loads(base_result)

    # 评估SFT模型
    sft_result = rl_tool.run({
        "action": "evaluate",
        "model_path": "./models/sft_full2",
        "max_samples": 100,
        "use_lora": True,
    })
    sft_data = json.loads(sft_result)

    # 对比结果
    print("模型对比:")
    print(f"预训练模型准确率: {base_data['accuracy']}")
    print(f"SFT模型准确率: {sft_data['accuracy']}")

if __name__ == "__main__":
    # t01()
    # t02()
    # t03()
    # t05()
    # t10()
    # t11()
    models = [None, "sft_model", "sft_minimal", "sft_practical", "sft_standard", "sft_memory_opt", "sft_full"]
    # for m in models: t12(m)
    t12("sft_full2")
    # t15()
    # t21()
    t22()
    pass