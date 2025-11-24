import difflib, os, re, json, requests, time

from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from tasks import LC303, LC11, LC198, LC65, LC862, LC3225
from prompts_cn_v2 import SYSTEM_PROMPT, INITIAL_PROMPT_TEMPLATE, REFLECT_PROMPT_TEMPLATE, REFINE_PROMPT_TEMPLATE

load_dotenv(dotenv_path='.env', override=True)
print(f'LLM_BASE_URL={os.getenv("LLM_BASE_URL")}')

def extract_block(text: str, tag: str) -> str:
    """
    提取 <tag></tag> 标签内容，用于调试查看模型思路
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

# --- 模块 1: 记忆模块 ---
class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """
    def __init__(self):
        # 初始化一个空列表来存储所有记录
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录。

        参数:
        - record_type (str): 记录的类型 ('execution' 或 'reflection')。
        - content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。
        """
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。
        """
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一轮尝试 (代码) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """
        获取最近一次的执行结果 (例如，最新生成的代码)。
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None

class Watchdog:
    def __init__(self, threshold=0.85, max_consecutive_loops=3):
        self.threshold = threshold
        self.max_consecutive_loops = max_consecutive_loops
        self.history_lines = []
        self.consecutive_loops = 0

    def is_looping(self, new_text_chunk):
        """
        检测新生成的文本是否在重复之前的废话。
        """
        # 简单的按行分割，实际使用中可能需要累积 buffer
        lines = new_text_chunk.strip().split('\n')
        
        for line in lines:
            if not line.strip():
                continue
                
            # 检查与上一行或上上行的相似度
            if self.history_lines:
                last_line = self.history_lines[-1]
                # 使用 SequenceMatcher 计算相似度
                similarity = difflib.SequenceMatcher(None, line, last_line).ratio()
                
                if similarity > self.threshold:
                    self.consecutive_loops += 1
                else:
                    self.consecutive_loops = 0 # 重置
            
            self.history_lines.append(line)
            
            # 保持历史记录不要太长
            if len(self.history_lines) > 20:
                self.history_lines.pop(0)
                
            if self.consecutive_loops >= self.max_consecutive_loops:
                return True
                
        return False

# --- 模块 2: Reflection 智能体 ---
class ReflectionAgent:
    def __init__(self, max_iterations=3):
        self.baseUrl = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
        self.apiKey = os.getenv("LLM_API_KEY", "dumy_api_key")
        self.model = os.getenv("LLM_MODEL_ID", "AUTO")
        self.timeout = int(os.getenv("LLM_TIMEOUT", 60))
        if self.model == "AUTO":
            self.model = self._figure_model()

        if not all([self.model, self.apiKey, self.baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        print(f"🛠️ 初始化 LLM 客户端，模型: {self.model}, 地址: {self.baseUrl}")
        self.client = OpenAI(api_key=self.apiKey, base_url=self.baseUrl, timeout=self.timeout)
        self.memory = Memory()
        self.max_iterations = max_iterations

    def _figure_model(self) -> str:
        response = requests.get(f'{self.baseUrl}/models', headers={'Authorization': f'Bearer {self.apiKey}'})
        response.raise_for_status()
        models = response.json().get('data', [])
        # 简单选择第一个模型，实际使用中可以根据需求选择
        if not models:
            raise ValueError("未能获取可用模型列表。")
        return models[0]['id']

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        watchdog = Watchdog(threshold=0.9, max_consecutive_loops=5)
        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        # initial_code = self._think(initial_prompt)
        # brainstorming 让模型进行发散型思考，temperature 设高一点
        initial_code = self._think(initial_prompt, temperature=0.5)
        initial_code = extract_block(initial_code, "code")
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环：反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._think(reflect_prompt, temperature=0.1)
            feedback = extract_block(feedback, "feedback")
            self.memory.add_record("reflection", feedback)

            if not feedback:
                print("❌ 无法解析反馈，结束迭代。")
                break

            # b. 检查是否需要停止
            if "无需改进" in feedback or "done" in feedback.lower():
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._think(refine_prompt, temperature=0)
            refined_code = extract_block(refined_code, "code")
            self.memory.add_record("execution", refined_code)
        
        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

    def _think(self, prompt: str, temperature=0.1, max_tokens=65536, topK=30, topP=0.8) -> str:
        """一个辅助方法，用于调用LLM并获取完整的流式响应。"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

def main():
    # 1. 初始化 Reflection 智能体，设置最多迭代5轮
    agent = ReflectionAgent(max_iterations=5)

    # 2. 定义任务并运行智能体
    tasks = [LC303, LC11, LC198, LC65, LC862, LC3225]
    task = tasks[5]

    start = time.time()
    print(f'\n=== 提示词 ===')
    print(f'\n=== SYSTEM_PROMPT ===\n{SYSTEM_PROMPT}')
    print(f'\n=== INITIAL_PROMPT_TEMPLATE ===\n{INITIAL_PROMPT_TEMPLATE}')
    print(f'\n=== REFLECT_PROMPT_TEMPLATE ===\n{REFLECT_PROMPT_TEMPLATE}')
    print(f'\n=== REFINE_PROMPT_TEMPLATE ===\n{REFINE_PROMPT_TEMPLATE}')
    print(f'\n=== 任务开始时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))} ===')
    agent.run(task)
    end = time.time()
    print(f'\n=== 任务结束时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))} ===')
    print(f'\n=== 任务总耗时: {end - start:.2f} 秒 ===')

if __name__ == '__main__':
    main()
