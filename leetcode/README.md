### Introduction
使用 Reflection 范式来尝试解决leetcode hard level 的问题

### Setup
- pull TGI docker images
```bash
# text-generation-inference docker images
docker images | grep 'ghcr.io/huggingface/text-generation-inference'
ghcr.io/huggingface/text-generation-inference   latest                    b6abf111199c   5 days ago      17.3GB
ghcr.io/huggingface/text-generation-inference   3.3.6                     3f235b36e71a   2 months ago    16.9GB

# check the model inside tgi container
docker exec -ti tgi /bin/bash -c 'ps aux | grep "model-id"'
```

### To-do list
- [ ] 

