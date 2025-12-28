"""
GitHub MCP 服务示例

注意：需要设置环境变量
    Windows: $env:GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here"
    Linux/macOS: export GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here"
"""
import os
from hello_agents.tools import MCPTool

def t01():
    # 创建 GitHub MCP 工具
    github_tool = MCPTool(
        server_command=["npx", "-y", "@modelcontextprotocol/server-github"]
    )

    # 1. 列出可用工具
    print("📋 可用工具：")
    result = github_tool.run({"action": "list_tools"})
    print(result)
    # 2. 搜索仓库
    print("\n🔍 搜索仓库：")
    result = github_tool.run({
        "action": "call_tool",
        "tool_name": "search_repositories",
        "arguments": {
            "query": "AI agents language:python",
            "page": 1,
            "perPage": 3
        }
    })
    print(result)

def t02():
    token = os.environ['GITHUB_PERSONAL_ACCESS_TOKEN']
    fetch_tool = MCPTool(
        server_command=["npx", "-y", "@modelcontextprotocol/server-fetch"]
    )

    print("📋 可用工具：")
    result = fetch_tool.run({"action": "list_tools"})
    print(result)

    # result = fetch_tool.run({
    #     "action": "call_tool",
    #     "tool_name": "search_repositories",
    #     "arguments":
    #     "url": "https://api.github.com/harrysun2006/repos",
    #     "method": "GET",
    #     "headers": {
    #         "Authorization": f"Bearer {token}",
    #         "Accept": "application/vnd.github+json"
    #     },
    #     "query": {
    #         "per_page": 100,
    #         "sort": "updated"
    #     }
    # })
    # print(result)

if __name__ == "__main__":
    # t01()
    t02()
    pass