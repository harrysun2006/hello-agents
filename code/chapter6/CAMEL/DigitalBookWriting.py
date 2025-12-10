from colorama import Fore
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.societies import RolePlaying
from camel.utils import print_text_animated

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-Coder-7B",
    api_key="dummy",
    url="http://192.168.18.77:8080/v1",
    model_config_dict={
        "temperature": 0.4,
        "max_tokens": 1024,
    },
)

def test():
    agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=model,
        token_limit=4096,
    )

    resp = agent.step("请用英文写一首关于noodle 的十四行诗，并在末尾就押韵，音节等规范加以对照说明")
    print(resp.msgs[0].content)

def write_book():
    # 定义协作任务
    task_prompt = """
创作一本关于"拖延症心理学"的短篇电子书，目标读者是对心理学感兴趣的普通大众。
要求：
1. 内容科学严谨，基于实证研究
2. 语言通俗易懂，避免过多专业术语
3. 包含实用的改善建议和案例分析
4. 篇幅控制在3000-5000字
5. 结构清晰，包含引言、核心章节和总结
6. 使用中文
"""

    print(Fore.YELLOW + f"协作任务:\n{task_prompt}\n")

    # 初始化角色扮演会话
    role_play_session = RolePlaying(
        model=model,
        assistant_role_name="心理学家", 
        user_role_name="作家", 
        task_prompt=task_prompt
    )

    print(Fore.CYAN + f"具体任务描述:\n{role_play_session.task_prompt}\n")

    # 开始协作对话
    chat_turn_limit, n = 30, 0
    input_msg = role_play_session.init_chat()

    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)
        
        print_text_animated(Fore.BLUE + f"作家:\n\n{user_response.msg.content}\n")
        print_text_animated(Fore.GREEN + f"心理学家:\n\n{assistant_response.msg.content}\n")
        
        # 检查任务完成标志
        if "CAMEL_TASK_DONE" in user_response.msg.content:
            print(Fore.MAGENTA + "✅ 电子书创作完成！")
            break
        
        input_msg = assistant_response.msg

    print(Fore.YELLOW + f"总共进行了 {n} 轮协作对话")

def main():
    # test()
    write_book()

if __name__ == "__main__":
    main()