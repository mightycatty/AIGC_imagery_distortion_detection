# -*- coding: utf-8 -*-
"""
@Time ： 2025/3/10 11:43
@Auth ： heshuai.sec@gmail.com
@File ：florence2_caption.py
"""
import requests
from tqdm import tqdm
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype,
                                             trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)


def run_example(img_path_or_url, task_prompt="<CAPTION>", text_input=None):
    if img_path_or_url.startswith("http"):
        image = Image.open(requests.get(img_path_or_url, stream=True).raw)
    else:
        image = Image.open(img_path_or_url)
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt,
                                                      image_size=(image.width, image.height))

    print(parsed_answer)


def main(img_root=None):
    if img_root is None:
        img_root = './data'
    img_list = [os.path.join(img_root, img) for img in os.listdir(img_root)]
