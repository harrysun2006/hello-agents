local(cpu), 20.4s
/Users/harry/micromamba/envs/v3.12/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Using device: cpu
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
模型和分词器加载完成！
编码后的输入文本:
{'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198, 108386,  37945, 100157, 107828,
           1773, 151645,    198, 151644,  77091,    198]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

模型的回答:
我是来自阿里云的大规模语言模型，我叫通义千问。我具有强大的自然语言处理能力，可以回答各种问题、创作文字，还能表达观点、撰写代码。

===
mayhu(cuda), 5s
/home/harry/micromamba/envs/v3.12/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Using device: cuda
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
模型和分词器加载完成！
编码后的输入文本:
{'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198, 108386,  37945, 100157, 107828,
           1773, 151645,    198, 151644,  77091,    198]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:0')}

模型的回答:
我是来自阿里云的大规模语言模型，我叫通义千问。我是一个能够回答问题、创作文字，还能表达观点、撰写代码的超大规模语言模型。我被设计用来帮助人们获取信息、完成任务和进行对话等。