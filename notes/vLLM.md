### Deploy/Run models locally
- huggingface + docker
```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
  -e HF_TOKEN=hf_xxx \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id Qwen/Qwen3-0.6B

docker run --gpus all --shm-size 1g -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id Qwen/Qwen3-0.6B

docker run --gpus all --shm-size 4g -p 8080:80 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id google/gemma-3-4b-it
```

- vllm
```bash
# 会和pytorch-gpu 冲突
# micromamba install vllm
pip install vllm==0.11.1
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype float16 \
  --gpu-memory-utilization 0.85

# vllm serve Qwen/Qwen3-0.6B --enable-reasoning --reasoning-parser deepseek_r1
vllm serve --host 0.0.0.0 --port 8000 --model Qwen/Qwen3-0.6B --dtype float16 --gpu-memory-utilization 0.85
vllm serve --host 0.0.0.0 --port 8000 --model google/gemma-3-4b-it --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 --swap-space 8
```

- curl test
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "用中文简单介绍一下 Qwen3-0.6B 是什么模型"}]
  }'

# 仅支持huggingface TGI
curl http://localhost:8080/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "用中文简单介绍一下 Gemma-3-4B-IT 是什么模型",
    "parameters": {
      "max_new_tokens": 2048
    }
  }'

curl http://localhost:8000/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-4b-it",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant that writes Python code."
      },
      {
        "role": "user",
        "content": "写一个快速排序算法"
      }
    ],
    "parameters": {
      "max_new_tokens": 4096
    }
  }'
```

- test in postman
user prompt(question): `写一个快速排序算法`
Ollama: http://192.168.18.77:11434/v1/chat/completions
hugging face (inference, docker): http://192.168.18.77:8080/v1/chat/completions
vLLM: http://192.168.18.77:8000/v1/chat/completions
vllm serve --host 0.0.0.0 --port 8000 --model google/gemma-3-4b-it --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 --swap-space 8

| server        | model                 | time (seconds)    |
|---------------|-----------------------|-------------------|
| Ollama        | qwen3:0.6b            | 12.6              |
| Ollama        | gemma3:4b             | 36 ~ 57           |
| TGI           | Qwen/Qwen3-0.6B       | 5.5               |
| TGI           | google/gemma-3-4b-it  | 16.3              |
| vLLM          | Qwen/Qwen3-0.6B       | 5.5               |
| vLLM          | google/gemma-3-4b-it  | 20.2              |
Ollama is slower than hugging face TGI & vLLM
time spent: Ollama ~= HF TGI x 2, HF TGI ~= vLLM

- ollama not using 100% GPU!
```bash
$ ollama ps
NAME          ID              SIZE      PROCESSOR          CONTEXT    UNTIL
qwen3:0.6b    7df6b6e09427    1.1 GB    85%/15% CPU/GPU    4096       3 minutes from now
```

- hf cmds
```bash
hf auth login
hf auth whoami
# need to grant access (agree T&C) at https://huggingface.co/google/gemma-3-4b-it
# need to explicitly grant permission in https://huggingface.co/settings/tokens
hf download google/gemma-3-4b-it --include="*"
# vllm 使用 HF 来下载 models & tokenizer!
```

