import torch
from PIL import Image
from cookiee.models import *
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor

model_path = "saves/cookiee-vlm/sft-long-caption"
image_path = "run_vlm/1.jpg"

generation_config = {
    "do_sample": False,
    "top_p": 0.95,
    "temperature": 0.7,
    "max_new_tokens": 2048,
}


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_path)
chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

model = model.to("cuda")

while True:
    text = input("Enter a prompt: ")
    if text == "q":
        break
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                #{"type": "text", "text": f"<|vision_start|><|image_pad|><|vision_end|>{text}"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    inputs_text = processor.apply_chat_template(messages, add_generation_prompt=True, chat_template=chat_template)
    print(inputs_text)
    input_ids = processor(images=Image.open(image_path), text=inputs_text, return_tensors="pt")
    input_ids = {k: v.to("cuda") for k, v in input_ids.items()}

    outputs = model.generate(**input_ids, **generation_config, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(outputs[0][input_ids["input_ids"].shape[1]:]))