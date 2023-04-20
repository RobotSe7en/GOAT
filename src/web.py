# -*- coding: utf-8 -*-
# Author: D.W.(wangpengtt@126.com)
# Time: 2023/04/10

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pathlib
import torch
import gradio as gr
import markdown as md
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, TextIteratorStreamer, TextStreamer
from threading import Thread
from peft import PeftModel
from loguru import logger


here = pathlib.Path(__file__).resolve().parent

device = 'cuda:0'
model_path = here.joinpath('../llama-30b')
lora_model_path = here.joinpath('../models/GOAT_sg30k_30B_001_LoRA')
logo_path = here.joinpath('../imgs/logo_new.png')

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

    tokenizer = LlamaTokenizer.from_pretrained(lora_model_path)
    add_token = "</s>"
    tokenizer.add_special_tokens({
        "eos_token": add_token,
        "bos_token": add_token,
        "unk_token": add_token,
    })
    tokenizer.bos_token_id=1
    model.eval()

    return model, tokenizer

# 2023.04.04 用于有监督训练数据的处理
def generate_prompt(text):
    '''
    单轮prompt
    TODO: 多轮对话
    '''
    text = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.</s>Human: {text}</s>Assistant: "
    return text
    

def chat(text, temperature, top_p, top_k, num_beams, max_new_tokens, repetition_penalty):
    '''
    聊天
    TODO: streaming
    '''
    logger.info(f'Input: {text}')
    prompt = generate_prompt(text)
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    if num_beams==1:
        # 使用transformers的TextIteratorStreamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        generation_config = dict(
            input_ids=input_ids,
            streamer=streamer,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            repetition_penalty=float(repetition_penalty)
        )
        thread = Thread(target=model.generate, kwargs=generation_config)
        thread.start()
        preds = ''
        for idx, new_text in enumerate(streamer):
            preds += new_text
            if idx % 3 == 0:  
                # print(md.markdown(f'{preds}.'))
                # yield md.markdown(f'{preds}.')
                yield f'{preds}.'
            elif idx % 3 == 1:
                # print(md.markdown(f'{preds}..'))
                # yield md.markdown(f'{preds}..')
                yield f'{preds}..'
            elif idx % 3 == 2:
                # print(md.markdown(f'{preds}..'))
                # yield md.markdown(f'{preds}...')
                yield f'{preds}...'
        # print(md.markdown(preds.replace('</s>', '')))
        # yield md.markdown(preds.replace('</s>', ''))
        yield preds.replace('</s>', '').replace('�', '')
        logger.info(f'Output: {preds}')
    else:
        # 非流式输出或使用transformers的TextStreamer
        streamer = TextStreamer(tokenizer)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            repetition_penalty=float(repetition_penalty)
        )
        with torch.no_grad():
            preds = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                do_sample=False,
                
            )
        
        output = tokenizer.batch_decode(preds)
        output = output[0].split('</s>')[2].replace('Assistant: ', '').replace('�', '').strip()
        logger.info(f'Output: {output}')
        yield output

def demo():
    '''
    web ui
    '''
    # refer to https://github.com/Facico/Chinese-Vicuna
    inputs = gr.components.Textbox(
        lines=2, label='输入', placeholder='请输入文本...'
    )
    temperature = gr.components.Slider(
        minimum=0.1, maximum=1.0, step=0.1, value=0.4, label='Temperature'
    )
    top_p = gr.components.Slider(
        minimum=0, maximum=0.95, step=0.01, value=0.75, label='top_p'
    )
    top_k = gr.components.Slider(minimum=0, maximum=50, step=1, value=40, label='top_k')
    num_beams = gr.components.Slider(
        minimum=1, maximum=5, step=1, value=1, label='num_beams'
    )
    max_new_tokens = gr.components.Slider(
        minimum=1, maximum=512, step=1, value=256, label='max_new_tokens'
    )
    repetition_penalty = gr.components.Slider(
        minimum=0.1, maximum=10.0, step=0.1, value=1.1, label='repetition_penalty'
    )
    
    description = f'''###### <center>GOAT(山羊)是中英文百亿参数大语言模型，采用LoRA方法以较低的资源在多轮对话数据集上SFT完成的。</center>'''

    gr.Interface(
        fn=chat,
        inputs=[inputs, temperature, top_p, top_k, num_beams, max_new_tokens, 
                repetition_penalty],
        outputs=[
            gr.Markdown(
                label="输出",
            )
        ],
        title="GOAT(山羊)大语言模型",
        description=description,
        allow_flagging='never'
    ).queue().launch(share=False, favicon_path=logo_path, server_name='0.0.0.0')

if __name__ == '__main__':
    logger.info(f'初始化模型...')
    model, tokenizer = init_model()
    logger.info(f'初始化模型完成.')
    demo()
