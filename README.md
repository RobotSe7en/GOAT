# <center>GOAT</center>

<div align="center">
    <img src='./imgs/logo.png' width=30%/>
</div>


GOAT(山羊)是中英文大语言模型，采用[LoRA](https://arxiv.org/pdf/2106.09685.pdf)方法以较低的资源基于[Llama](https://github.com/facebookresearch/llama)在多轮对话数据集上SFT。本项目下的代码、数据、模型等只供研究使用。(logo由[文心一言](https://yiyan.baidu.com/)生成)

## 更新
### 🚀 2023.04.21
- [x] 🎉发布了30B和13B的LoRA参数，此参数基于shareGPT的30k数据SFT,epoch=2；
- [x] 🎉web页面增加了流式输出； 
- [x] 🎉使用了`transformer==4.28.1`，支持`num_beams=1`时流式输出；
- [x] 🎉添加了演示视频。


### 🚀 2023.04.15
- [x] 🎉增加了处理多轮对话类数据的代码，使得代码可以对LlaMa模型进行多轮对话有监督微调，多轮对话有监督微调的模型效果具有较大提升；
- [x] 🎉将`'### '`和`'\n\n'`切分符替换成了eos_token`'</s>'`，使其能够更好的识别文本的段落和角色的切分。模型在推理时能够适时地结束文本生成而不是无休止的生成；
- [x] 🎉公开GOAT_7B LoRA模型参数，此模型是基于LlaMa在10k中英文多轮对话数据上有监督微调获得；
- [x] 🎉多轮对话数据示例可在[sample.json](./datasets/sample.json)查看，完整的shareGPT多轮对话数据可在[这里](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)下载。

## 效果
https://user-images.githubusercontent.com/14015706/233425087-dec0d125-b2e7-4fc5-85b6-cadd23083fb2.mp4


## 模型
本项目是基于[Llama](https://github.com/facebookresearch/llama)使用多轮对话数据集SFT的模型，使用本项目代码进行微调或推理需要先[申请](https://github.com/facebookresearch/llama)或在[Huggingface](https://huggingface.co/models)下载Llama原模型权重并放在对应文件夹下。

## 微调参数
本项目在1台RTX A6000(48G)显卡上训练了2个epoch，batch_size是128：
```
    max_lenght=1024
    per_device_train_batch_size=8
    gradient_accumulation_steps=16
    learning_rate=3e-4
```

## 局限性
 - 由于Llama只有少部分中文token、没有在中文语料下预训练、微调数据量较少、只更新了Adapter参数等种种因素导致微调之后的中文效果不是非常理想；
 - 会出现较多的事实性错误；
 - 会出现“复读机”情况；
 - 产生偏见、危险、政治错误等言论。

## TODO
 - [x] 在对话类数据集上使用LoRA进行微调；
 - [x] 提供web页面，并支持流式输出；
 - [x] 微调LlaMa 30B模型；
 - [ ] 实现多轮对话；
 - [ ] 实现基于知识库或文本语料的问答(LangChain或自己构建)；
 - [ ] 使用RLHF；
 - [ ] 重构代码使其可用于多卡并行训练；
 - [ ] ...
