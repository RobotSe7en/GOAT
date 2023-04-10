# -*- coding: utf-8 -*-
# Author: D.W.(wangpengtt@126.com)
# Time: 2023/04/10

import pathlib
import torch
import gradio as gr
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel
from loguru import logger

here = pathlib.Path(__file__).resolve().parent

device = 'cuda:0'
model_path = here.joinpath('../llama-7b')
lora_model_path = here.joinpath('../models/GOAT_001')

def init_model():
    '''
    初始化model和tokenizer
    '''
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    model = PeftModel.from_pretrained(
        model, 
        lora_model_path,
        torch_dtype=torch.float16,
        device_map={'':0}
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model.eval()

    return model, tokenizer

# 2023.04.04 用于有监督训练数据的处理
def generate_prompt(text):
    '''
    单轮prompt
    TODO: 多轮对话
    '''
    text = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n### Human: {text}\n\n### Assistant: "
    return text
    

def chat(text, temperature, top_p, top_k, num_beams, max_new_tokens, repetition_penalty):
    '''
    聊天
    TODO: streaming
    '''
    logger.info(f'Input: {text}')
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        repetition_penalty=float(repetition_penalty)
    )
    prompt = generate_prompt(text)
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    with torch.no_grad():
        preds = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )
    output = tokenizer.batch_decode(preds)
    output = output[0].split('### ')[2].replace('Assistant: ', '').replace('�', '').strip()
    logger.info(f'Output: {output}')
    return output

def demo():
    '''
    web ui
    '''
    # refer to https://github.com/Facico/Chinese-Vicuna
    inputs = gr.components.Textbox(
        lines=2, label='输入', placeholder='请输入文本...'
    )
    temperature = gr.components.Slider(
        minimum=0, maximum=1.0, step=0.1, value=0.4, label='Temperature'
    )
    top_p = gr.components.Slider(
        minimum=0, maximum=0.95, step=0.01, value=0.75, label='top_p'
    )
    top_k = gr.components.Slider(minimum=0, maximum=50, step=1, value=40, label='top_k')
    num_beams = gr.components.Slider(
        minimum=1, maximum=5, step=1, value=4, label='num_beams'
    )
    max_new_tokens = gr.components.Slider(
        minimum=1, maximum=512, step=1, value=256, label='max_new_tokens'
    )
    repetition_penalty = gr.components.Slider(
        minimum=0.1, maximum=10.0, step=0.1, value=2.0, label='repetition_penalty'
    )
    
    gr.Interface(
        fn=chat,
        inputs=[inputs, temperature, top_p, top_k, num_beams, max_new_tokens, 
                repetition_penalty],
        outputs=[
            gr.inputs.Textbox(
                lines=25,
                label="输出",
            )
        ],
        title="GOAT(山羊)中英文大语言模型",
        description="GOAT(山羊)是中英文大语言模型，采用LoRA方法以较低的资源基于Llama在50k的中英文数据集上指令微调。本项目下的代码、数据、模型等只供研究使用。",
    ).queue().launch(share=False)

if __name__ == '__main__':
    logger.info(f'初始化模型...')
    model, tokenizer = init_model()
    logger.info(f'初始化模型完成.')
    demo()
