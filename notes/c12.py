### chapter 12

import os

import pandas as pd
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download
from hello_agents.evaluation import BFCLDataset, GAIADataset
from dotenv import load_dotenv

load_dotenv(override=True)

### 12.2.2 获取 BFCL 数据集

# 方法 2：使用 HelloAgents 加载官方数据
def t01():
    # 加载BFCL官方数据
    dataset = BFCLDataset(
        bfcl_data_dir="./temp_gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
        category="simple_python"  # BFCL v4类别
    )

    # 加载数据（包括测试数据和ground truth）
    data = dataset.load()

    print(f"✅ 加载了 {len(data)} 个测试样本")
    print(f"✅ 加载了 {len(dataset.ground_truth)} 个ground truth")
    # 输出:
    # ✅ 加载了 400 个测试样本
    # ✅ 加载了 400 个ground truth

    # 获取所有支持的类别
    categories = dataset.get_available_categories()
    print(f"支持的类别: {categories}")

def to_jsonl(parquet_path):
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        print(f"[WARN] Parquet file does not exist: {parquet_path}")
        return None
    
    if parquet_path.suffix != ".parquet":
        print(f"[WARN] Not a parquet file: {parquet_path}")
        return None

    jsonl_path = parquet_path.with_suffix(".jsonl")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[WARN] Failed to read parquet file: {parquet_path}")
        print(f"       Reason: {e}")
        return None

    try:
        df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
        return jsonl_path
    except Exception as e:
        print(f"[WARN] Failed to write jsonl file: {jsonl_path}")
        print(f"       Reason: {e}")
        return None

### 12.3.2 获取 GAIA 数据集
# datasets==4.4.2; huggingface_hub==1.2.3
# 注意 item.get("Level") 是 str 类型!
def t02():
    # 使用 huggingface dataset
    data_dir = snapshot_download(
        repo_id="gaia-benchmark/GAIA", 
        repo_type="dataset",
        local_dir="./data/gaia")
    # dataset = load_dataset(data_dir, "2023_level1", split="test")
    dataset = load_dataset(data_dir, "2023_all", split="test[:10]")
    print(type(dataset))
    for example in dataset.data.to_pylist()[:3]:
        question = example["Question"]
        file_path = os.path.join(data_dir, example["file_path"])
        print(f"{question} ==> {file_path}")
    
    # 使用 helloagent 类
    # GAIADataset 目前只支持 .jsonl 文件
    to_jsonl("./data/gaia/2023/test/metadata.parquet")
    to_jsonl("./data/gaia/2023/validation/metadata.parquet")
    
    dataset = GAIADataset(
        dataset_name="gaia-benchmark/GAIA",
        split="validation",  # 或 "test"
        level=1  # 可选: 1, 2, 3, None(全部)
    )
    items = dataset.load()
    print(f"加载了 {len(items)} 个测试样本")

    test_ds = GAIADataset(split="test")
    stats = test_ds.get_statistics()
    print(f"test总样本数: {stats['total_samples']}")
    print(f"test级别分布: {stats['level_distribution']}")
    val_ds = GAIADataset()
    stats = val_ds.get_statistics()
    print(f"val总样本数: {stats['total_samples']}")
    print(f"val级别分布: {stats['level_distribution']}")

if __name__ == "__main__":
    # t01()
    # to_jsonl("./data/gaia/2023/test/metadata.parquet")
    # to_jsonl("./data/gaia/2023/validation/metadata.parquet")
    t02()
    pass