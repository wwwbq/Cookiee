import torch
from PIL import Image
from cookiee.models import *
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor

model_path = "saves/prefer-clip-qwen2.5/sft-long-caption-cleaned"
image_path = "run_vlm/1.jpg"

generation_config = {
    "do_sample": True,
    "top_p": 0.95,
    "temperature": 0.7,
    "max_new_tokens": 2048,
}


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_path)

model = model.to("cuda")

while True:
    text = input("Enter a prompt: ")
    if text == "q":
        break
    
    messages = [{"role": "user", "content": f"{processor.image_token}{text}"}]
    inputs_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    input_ids = processor(images=Image.open(image_path), text=inputs_text, return_tensors="pt")
    input_ids = {k: v.to("cuda") for k, v in input_ids.items()}

    outputs = model.generate(**input_ids, **generation_config, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(outputs[0][input_ids["input_ids"].shape[1]:]))
