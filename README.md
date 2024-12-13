# 从零开始构建一个大型语言模型
本文是使用PyTorch从0开始逐步实现一个类似ChatGPT那样的大型语言模型（以下简称LLM），步骤大体分为如下：
  - 数据准备与预处理
  - 模型架构与实现
  - 模型训练与评估
  - 文本生成与微调

## 前言

如果你让ChatGPT继续输出“每一次努力都让你感动”后面的内容，ChatGPT会自动延续后续内容并输出，我第一次看到就对此产生了好奇。本文是实现一个类似这样的类GPT模型，且针对文本数据，力求尽量不出现公式，尽量减少专业术语，使用简单清晰的词语进行解释，并给出代码实现。初步可能为了更好的理解，实现一个简单的版本，它可能输出的并不理想，但最终会调整为适合现有大模型的复杂实现思路。故你可以将本文的目的理解为新手入门LLM通识训练。
废话不多说，现在开始！

首先，我们要先明确我们此行的目的：给一段输入“每一次努力都让你感动”，经过模型的处理加工后，输出后续的内容“未来的精彩由此慢慢绽放”。用程序化的语言可以这么表达：输入input，经过model的处理，输出output。一般情况下，模型是依据历史文本来生成固定长度的内容，不可能无限制的生成，所以还需要限定输入长度和输出长度，由此我们定义一个简单的文本生成方法 `generate_text_simple`:

``` python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    return idx
```

这个方法就是最终我们需要输出新内容所需要的，我们对这个方法进行解释：
  - `model` : 用来生成文本的大规模语言模型。这个模型经过训练，可以根据给定的输入（上下文）预测并生成接下来最有可能出现的文本。
  - `idx` : 这通常指代的是输入文本序列中每个词或标记（token）在词汇表中的索引位置。在处理过程中，文本首先会被分词器转换成一系列的标记，然后每个标记会根据词汇表映射为一个索引值，用于模型的计算。
  - `max_new_tokens` : 这是设定的一个上限，表示模型在生成新文本时最多可以输出的新标记数量。它有助于控制生成文本的长度，避免生成过长的内容。
  - `context_size` : 模型在做预测时所参考的历史信息长度，即每次提供给模型的输入序列的长度。较大的上下文大小可以让模型记住更多过去的信息，从而可能生成更加连贯和有意义的文本。

我们试着简单的实现这个方法：
- 首先定义 `model` 参数的类型，名字为 `GPTModel`, 因为我们不知道如何实现它，姑且暂时将输入作为输出返回:

``` python
import torch
import torch.nn as nn
print("torch version:",torch.__version__)

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_idx):
        return in_idx

```

- 然后实现 `generate_text_simple` 方法:

``` python
import random
def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        # 获取上下文token
        idx_cond = idx[-context_size:] 
        # 通过模型生成后续序列
        logits = model(idx_cond) 
        # 随机选择一个
        idx_next = random.choices(logits)[0] 
        #将下一个预测添加到序列中
        idx.append(idx_next) 
        
    return idx
```

- 最后我们进行测试，并且有了输出，最大续写5个字，每次只处理上下文4个长度：

``` python

model = GPTModel()
in_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
generate_text_simple(model=model, idx=in_idx, max_new_tokens=5, context_size=4)

```
输出

``` python

[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 13, 16, 16, 16, 16]

```

但这个结果人类看不懂，并且是很随机的输出，所以它毫无意义。因此我们需要将这个模型进行完善，分析以上过程和代码，我们需要解决的问题有：
- 将输入文本转为词汇表的索引位置
- 实现模型预测，既完成模型训练
- 选择最有可能或最优的预测，而不是随机输出
- 使用正常的文本来测试，且将结果输出为人类能看懂的文本