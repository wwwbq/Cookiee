from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List

from .base import BasePreprocessor
from ..mm_plugin import BasePlugin
from ...constants import IGNORE_INDEX
from ...configs import DatasetArguments

from transformers import PreTrainedTokenizer

from helper import get_logger


logger = get_logger("sft-preprocessor")


class SftPreprocessor(BasePreprocessor):
    def process(
        self,
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizer,
        mm_plugin: BasePlugin = None,
        dataset_args: DatasetArguments = None,
        chat_template=None,
    ):
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.

        model_inputs = defaultdict(list)

        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning("Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i]))
                continue
            
            system = examples["_system"][i]
            prompt = examples["_prompt"][i]
            response = examples["_response"][i]

            if system is not None:
                if isinstance(system, str):
                    system = [{"role": "system", "content": system}]
                prompt = system + prompt
            
            if mm_plugin is not None:
                prompt = mm_plugin.preprocess_multi_modal_messages(prompt, images=examples["_images"][i])

            prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, chat_template=chat_template)
            label_ids = tokenizer(response[0]["content"], add_special_tokens=False).input_ids

            prompt_len, label_len = self.truncate_sequence_length(len(prompt_ids), len(label_ids), dataset_args.max_seq_length)

            prompt_ids = prompt_ids[:prompt_len]
            if dataset_args.use_bos_token:
                prompt_ids = [tokenizer.bos_token_id] + prompt_ids

            label_ids = label_ids[:label_len]
            if dataset_args.use_eos_token:
                label_ids += [tokenizer.eos_token_id]

            input_ids = prompt_ids + label_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + label_ids

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["length"].append(len(input_ids))

        return model_inputs


    def __call__(
        self,
        examples: Dict[str, List[Any]],
        tokenizer: "PreTrainedTokenizer",
        mm_plugin: BasePlugin = None,
        dataset_args: DatasetArguments = None,
        chat_template=None,
        *args,
        **kwargs
    ):
        if not hasattr(self, "packing") and dataset_args.packing:
            logger.warning(f"Packing does not support for {self.__class__.__name__}, using vanilla process instead.")
            dataset_args.packing = False

        if dataset_args.packing:
            model_inputs = self.packing(examples, tokenizer, mm_plugin, dataset_args, chat_template, *args, **kwargs)
        else:
            model_inputs = self.process(examples, tokenizer, mm_plugin, dataset_args, chat_template, *args, **kwargs)

        model_inputs["images"] = examples["_images"]
        model_inputs["channel"] = examples["_channel"]

        return model_inputs 


    @staticmethod
    def truncate_sequence_length(prompt_len, label_len, max_seq_length):
        # 如果token比较短，那么保证label尽量完整，truncate prompt
        if label_len * 2 < max_seq_length:
            max_new_tokens = max_seq_length
        # 如果prompt比较短，那么保证prompt完整，truncate label
        elif prompt_len * 2 < max_seq_length:
            max_new_tokens = max_seq_length - prompt_len
        # 二者都比较长，按比例截断
        else:
            max_new_tokens = int(max_seq_length * (label_len / (label_len + prompt_len)))

        label_len = min(max_new_tokens, label_len)
        max_prompt_len = max(max_seq_length - label_len, 0)
        prompt_len = min(max_prompt_len, prompt_len)

        return prompt_len, label_len
    

    def print_example(self, example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer", mm_plugin=None) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("*"*50)
        print("input_ids:\n{}".format(example["input_ids"]))
        print("*"*50)
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("*"*50)
        print("label_ids:\n{}".format(example["labels"]))
        print("*"*50)
        print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))
        print("*"*50)
