import torch

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

from cookiee.models import *
from cookiee.data.template import VICUNA_CHAT_TEMPLATE

class LlavaVLM(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path, **kwargs):

        flash_attn_flag = False
        try:
            import flash_attn

            flash_attn_flag = True
        except ImportError:
            pass
        
        if flash_attn_flag:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=torch.bfloat16
            )

        self.processor = AutoProcessor.from_pretrained(model_path)

        model = model.eval()
        self.model = model.cuda()

        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=128, top_p=None, num_beams=1
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )


    def output_process(self, answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        elif "<|endoftext|>" in answer:
            answer = answer.split("<|endoftext|>")[0].strip()
        return answer


    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False


    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message


    def _generate(self, conversation, images=[]):
        prompt = self.processor.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(
            "cuda", torch.bfloat16
        )

        output = self.model.generate(**inputs, **self.kwargs, eos_token_id=self.processor.tokenizer.eos_token_id)

        answer = self.processor.decode(output[0], skip_special_tokens=True)
        answer = self.output_process(answer)
        answer = answer.replace('<unk>', '')

        return answer


    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += self.processor.image_token
                images.append(item["value"])
        return text, images


    def generate_inner(self, message, dataset=None):
        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert("RGB") for s in images]
        
        conversation = [{"role": "user", "content": content}]

        answer = self._generate(conversation, images)

        return answer
    

    def chat_inner(self, message, dataset=None):
        conversation, images = [], []
        for utter in message:
            content, images_sub = self.concat_tilist(utter["content"])
            if utter["role"] == "user":
                conversation.append({"role": "user", "content": content})
            else:
                conversation.append({"role": "assistant", "content": content})
            images.extend(images_sub)
            
        assert message[-1]["role"] == "user", message
        
        images = [Image.open(s).convert("RGB") for s in images]
        
        output = self._generate(conversation, images)
        
        return output
    

class CookieeVLM(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True
    chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    def __init__(self, model_path, **kwargs):

        flash_attn_flag = False
        try:
            import flash_attn

            flash_attn_flag = True
        except ImportError:
            pass
        
        if flash_attn_flag:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )

        self.processor = AutoProcessor.from_pretrained(model_path)

        model = model.eval()
        self.model = model.cuda()

        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=128, top_p=None, num_beams=1
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )


    def output_process(self, answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        elif "<|endoftext|>" in answer:
            answer = answer.split("<|endoftext|>")[0].strip()
        return answer


    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False


    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message


    def _generate(self, conversation, images=[]):
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, chat_template=self.chat_template
        )
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(
            "cuda", torch.bfloat16
        )

        output = self.model.generate(**inputs, **self.kwargs, eos_token_id=self.processor.tokenizer.eos_token_id)

        answer = self.processor.decode(output[0], skip_special_tokens=True)
        answer = self.output_process(answer)
        answer = answer.replace('<unk>', '')

        return answer


    def convert_one_turn_message_format(self, role, message):
        assert role in ["user", "assistant"]
        format_messages, images = {"role": role, "content": []}, []
        for item in message:
            # 单轮对话
            assert "type" in item and "value" in item
            if item["type"] == "text":
                format_messages["content"].append({"type": "text", "text": item["value"]})
            elif item["type"] == "image":
                format_messages["content"].append({"type": "image"})
                images.append(item["value"])
        return [format_messages], images
    

    def convert_muilt_turn_message_format(self, messages):
        format_messages, images = [], []
        for message in messages:
            assert "role" in message
            role, content = message["role"], message["content"]
            cur_turn_message, cur_turn_imgs = self.convert_one_turn_message_format(role, content)
            format_messages.extend(cur_turn_message)
            images.extend(cur_turn_imgs)
        return format_messages, images


    def generate_inner(self, message, dataset=None):
        # Support interleave text and image
        conversation, images = self.convert_one_turn_message_format("user", message)

        images = [Image.open(s).convert("RGB") for s in images]

        if len(images) == 0:
            images = None
            
        answer = self._generate(conversation, images)

        return answer
    

    def chat_inner(self, message, dataset=None):
        conversation, images = self.convert_muilt_turn_message_format(message)
            
        assert message[-1]["role"] == "user", message
        
        images = [Image.open(s).convert("RGB") for s in images]
        
        output = self._generate(conversation, images)
        
        return output