### Hugging Face Hub models local deployment
#### ✔ 方式 A：本地使用 `transformers` 加载模型（最灵活）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```
这是最常见的本地推理方式。

---

#### ✔ 方式 B（强烈推荐）：Docker + TGI（Text Generation Inference）
**TGI 是 Hugging Face 官方推荐的 LLM 服务器，性能强、API 标准化，并支持 OpenAI API 风格。**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# or sudo sv restart docker
# equivelant to below:
vi /etc/docker/daemon.json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}

# 验证 GPU 是否成功挂载
docker run --rm --runtime=nvidia --gpus all ubuntu:22.04 nvidia-smi
# docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

# run docker or via docker-compose to start up a TGI server loading hf models!
docker run --gpus all --shm-size 1g -p 8080:80 \
    -e HF_TOKEN=hf_xxx \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id Qwen/Qwen3-0.6B
```

然后你就得到一个：`http://localhost:8080/v1/chat/completions` ，完全兼容 OpenAI API！
还可以通过openai 访问API：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
```

---

#### ✔ 方式 C：使用 `llama.cpp`（适合 GGUF 量化模型）

很多模型提供 GGUF 版本（特别是大模型压缩用），可以用 llama.cpp：

```bash
llama.cpp/main -m ./model.gguf -p "hello"
```

也可以 Docker 运行：

```bash
docker run -p 8080:8080 \
    -v ./models:/models \
    ghcr.io/ggerganov/llama.cpp:latest \
    --model /models/Qwen2.5-7B.gguf

docker run -p 8080:8080 \
    -v ./models:/models \
    ghcr.io/ggerganov/llama.cpp:latest \
    --model /models/model.gguf --ctx-size 4096
```

它同样支持 OpenAI API 格式：`POST /v1/chat/completions`

---

#### ✔ 方式 D：使用 vLLM（速度极快、分布式支持）

如果你需要高吞吐推理（特别是服务端场景）：

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct

# or via docker
docker run --gpus all -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct
```

这也直接开启 OpenAI API：`curl http://localhost:8000/v1/chat/completions -d '{...}'`

---

## 🔥 总结（本地部署模型的四大主流方式）

| 部署方式              | 适用场景          | 是否支持 OpenAI API |
| ----------------- | ------------- | --------------- |
| transformers 本地加载 | 离线推理 / 研发     | ❌（除非你自己写）       |
| **TGI（最推荐）**      | 生产服务、GPU 推理   | ✅（官方支持）         |
| llama.cpp         | GGUF 量化、小显存设备 | ✅               |
| **vLLM（性能极高）**    | 大吞吐量服务        | ✅               |

> 三个推理服务器 **TGI / vLLM / llama.cpp** 都能“像 OpenAI 一样”开放接口，是业界最常用的 Hugging Face 私有化部署方案。

---

# 🧩 3. 你到底应该选择哪一种部署方式？

| 需求                      | 最佳方案                |
| ----------------------- | ------------------- |
| GPU 支持 + 大模型 + 企业生产稳定性  | **TGI**             |
| 高并发（>100 QPS） + 批量推理    | **vLLM**            |
| 小显存、在 CPU / Mac / 嵌入设备跑 | **llama.cpp（GGUF）** |
| 研发阶段、Notebook 调试        | transformers        |

如果你说一下：

* 你的 **GPU**（显存多少？）
* 模型大小（例如 7B / 13B / 70B？）
* 你需要 **OpenAI API 接口** 还是直接 Python？
* 是否要用于 **在线服务**？

我可以给你定制一份最适合你的“部署架构方案 + Docker 启动脚本”。
