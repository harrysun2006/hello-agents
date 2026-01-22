### chapter 11

import json, os, re
import torch
from datasets import Dataset
from hello_agents.rl import format_math_dataset
from hello_agents.tools import RLTrainingTool
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

## 11.2.3 自定义数据集和奖励函数

rl_tool = RLTrainingTool()
sft_dataset = None
rl_dataset = None

# 确保 torch 安装了GPU支持库，应优先使用 GPU!
def t00():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Found device: {device}, {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")

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

# SFT 训练
def t10():
    # 创建训练工具
    rl_tool = RLTrainingTool()

    # SFT训练
    config = {
        "action": "train",
        "algorithm": "sft",
        
        # 模型配置
        "model_name": "Qwen/Qwen3-0.6B",
        "output_dir": "./output/sft_standard",
        
        # 数据配置
        "max_samples": 1000,  # 使用1000个样本
        
        # 训练配置
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 5e-5,
        
        # LoRA配置
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
    }
    result = rl_tool.run(config)
    result_dict = json.loads(result)

    print(f"\n✓✓✓ 训练完成: {result_dict}")
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
    base = "Qwen/Qwen3-0.6B"
    if name is None: 
        model_name = base
    else:
        if Path(name).exists(): model_name = name
        elif Path(f"./models/{name}").exists(): model_name = f"./models/{name}"
        else: model_name = base

    print(f"✅ 使用模型{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base, fix_mistral_regex=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda")

    # 测试问题
    question = """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""

    # 构造输入
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成回答
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n✅ 模型{model_name}的回答:")
    print(response)

def t12b(name: str = None):
    base = "Qwen/Qwen3-0.6B"
    if name is None: 
        model_name = base
    else:
        if Path(name).exists(): model_name = name
        elif Path(f"./models/{name}").exists(): model_name = f"./models/{name}"
        else: model_name = base

    print(f"✅ 使用模型{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 测试问题
    question = """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""

    messages = [
        {"role": "user", "content": question}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    outputs = model.generate(inputs, max_new_tokens=300, eos_token_id=tokenizer.eos_token_id,)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

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

# TODO: 直接调用 trl 训练 GRPO 模型!
# GRPO 训练
def t20():
    tool = RLTrainingTool()

    config = {
        "action": "train",
        "algorithm": "grpo",
        
        # 模型配置 - 可以使用SFT训练后的模型
        "model_name": "Qwen/Qwen3-0.6B",  # 或 "./output/sft_standard"
        "output_dir": "./output/grpo_standard",
        
        # 数据配置
        "max_samples": 500,  # GRPO通常使用较少样本
        
        # 训练配置
        "num_epochs": 3,
        "batch_size": 2,  # GRPO需要更多显存
        "learning_rate": 1e-5,  # 比SFT小10倍
        
        # LoRA配置
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
    }
    
    print("标准GRPO训练配置:")
    print(f"  模型: {config['model_name']}")
    print(f"  样本数: {config['max_samples']}")
    print(f"  训练轮数: {config['num_epochs']}")
    print(f"  batch_size: {config['batch_size']}")
    print(f"  learning_rate: {config['learning_rate']} (比SFT小)")
    
    # 实际训练时取消注释
    result = tool.run(config)
    result_dict = json.loads(result)
    print(f"\n✅ GRPO训练完成! {result}")
    
    return config

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

# 对比不同阶段的模型, 理解 SFT, GRPO 的效果
# TODO: grpo_standard 重新训练后评估准确度依然为0!
def t22():
    # 评估GRPO模型
    grpo_result = rl_tool.run({
        "action": "evaluate",
        "model_path": "./models/grpo_standard",
        "max_samples": 100,
        "use_lora": True,
    })
    grpo_data = json.loads(grpo_result)

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
    print(f"GRPO模型准确率: {grpo_data['accuracy']}")

def t23(name: str = None):
    base = "Qwen/Qwen3-0.6B"
    if name is None: 
        model_name = base
    else:         
        if Path(name).exists(): model_name = name
        elif Path(f"./models/{name}").exists(): model_name = f"./models/{name}"
        else: model_name = base

    result = rl_tool.run({
        "action": "evaluate",
        "model_path": model_name,
        "max_samples": 100,
        "use_lora": True,
    })
    data = json.loads(result)

    # 对比结果
    print("模型对比:")
    print(f"模型 {model_name} 准确率: {data['accuracy']}")

if __name__ == "__main__":
    t00()
    # t01()
    # t02()
    # t03()
    # t05()
    # t10()
    # t11()
    models = [None, "sft_model", "sft_minimal", "sft_practical", "sft_standard", "sft_memory_opt", "sft_full"]
    # for m in models: t12(m)
    # t12("sft_full2")
    # t15()
    t20()
    # t12("./output/sft_standard")
    t12("./output/grpo_standard")
    # t12b("./output/grpo_standard")
    # t12("./models/sft_full2")
    # t12b("./models/sft_full2")
    # t21()
    # t22()
    # t23("./output/sft_standard")
    t23("./output/grpo_standard")
    pass