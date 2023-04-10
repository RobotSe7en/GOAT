# <center>GOAT</center>

<div align="center">
<img src='./imgs/logo.png' width=60%/>
</div>


GOAT(山羊)是中英文大语言模型，采用[LoRA](https://arxiv.org/pdf/2106.09685.pdf)方法以较低的资源基于[Llama](https://github.com/facebookresearch/llama)在50k的中英文数据集上指令微调。本项目下的代码、数据、模型等只供研究使用。

### 模型
本项目是基于[Llama](https://github.com/facebookresearch/llama)指令微调的模型，使用本项目代码进行微调或推理需要先[申请](https://github.com/facebookresearch/llama)或在[Huggingface](https://huggingface.co/models)下载Llama原模型权重。微调后的Adapter权重从[这里](https://huggingface.co/dannywong/GOAT)下载，并放在[GOAT_001_13B_Lora](./models/GOAT_001_13B_Lora/)目录下。

### 微调
本项目在1台RTX A6000(48G)显卡上训练了5个epoch，batch_size是128：
```
    max_lenght=512
    per_device_train_batch_size=32
    gradient_accumulation_steps=4
    learning_rate=3e-4
```

## TODO
 - 在对话类数据集上使用LoRA进行微调；
 - 在对话类数据集上进行全量微调；
 - 重构代码使其可用于多卡并行训练；
 - 基于Llama 30B和65B微调；
 - ...
