### Files
- [LLM.md](LLM.md): 主要 LLM 模型参数, 特点比较
- [MEMO.md](MEMO.md): 备课(上课)大纲
- [练习00-提示词](P00_prompts.md): 练习00(A/B提示词对比演示)问题及提示词
- [日志调试TGI Proxy](P00_proxy.py): 代理用来透传LLM请求响应, 方便日志调试。还有待完善
- [练习00](P00.md): 练习00备课及笔记

### Ollama
```bash
ollama pull qwen2:0.5b-instruct
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5:3b-instruct
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
ollama pull qwen3:4b
```

### `command` in docker-compose.yaml
```yaml
# 支持的几种写法:
  vllm1:
    ...
    command:
      - --model
      - Qwen/Qwen2.5-3B-Instruct
      - --host
      - 0.0.0.0
      - --port
      - "8000"

  vllm2:
    ...
    command: >
      --model Qwen/Qwen2.5-3B-Instruct
      --host 0.0.0.0 --port 8000

  vllm3:
    ...
    command: >
      --model=Qwen/Qwen2.5-3B-Instruct
      --host=0.0.0.0
      --port=8000
      --dtype=float16

```
