from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List
import collections
from collections import Counter

from .base import BasePreprocessor
from ..mm_plugin import BasePlugin
from ...configs import DatasetArguments


from transformers import PreTrainedTokenizer

from helper import get_logger


logger = get_logger("preference_pretrain-preprocessor")


class PreferencePretrainPreprocessor(BasePreprocessor):
    def process(
        self,
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizer,
        mm_plugin: BasePlugin = None,
        dataset_args: DatasetArguments = None,
        chat_template=None,
    ):
        bos_token = tokenizer.bos_token if dataset_args.use_bos_token else ""
        eos_token = tokenizer.eos_token if dataset_args.use_eos_token else ""

        chosen_examples = [bos_token + messages + eos_token for messages in examples["_chosen"]]
        rejected_examples = [bos_token + messages + eos_token for messages in examples["_rejected"]]

        if mm_plugin is not None:
            chosen_examples = [
                mm_plugin.preprocess_multi_modal_messages(
                    messages=chosen_examples[i],
                    images=examples["_images"][i],
                )
                for i in range(len(chosen_examples))
            ]
            rejected_examples = [
                mm_plugin.preprocess_multi_modal_messages(
                    messages=rejected_examples[i],
                    images=examples["_images"][i],
                )
                for i in range(len(rejected_examples))
            ]

        # include input_ids and attention_mask, attention_mask is always 1
        chosen_inputs = tokenizer(
            chosen_examples, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=dataset_args.max_seq_length
        )
        rejected_inputs = tokenizer(
            rejected_examples, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=dataset_args.max_seq_length
        )

        model_inputs = {}
        for key in ["input_ids", "attention_mask"]:
            model_inputs[f"chosen_{key}"] = chosen_inputs[key]
            model_inputs[f"rejected_{key}"] = rejected_inputs[key]

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

        # 暂时不支持packing，也不用支持packing
        model_inputs = self.process(examples, tokenizer, mm_plugin, dataset_args, chat_template, *args, **kwargs)
        model_inputs["domain"] = examples['_domain']
        model_inputs["images"] = examples["_images"]

        # 为了方便处理，只计算chosen的长度, 实际上chosen和rejected的长度差别不会很大
        model_inputs["length"] = [len(ids) for ids in model_inputs["chosen_input_ids"]]
        
        # 文本的长度为负值
        model_inputs["modality_lengths"] = [
            -len(ids) if img is None else len(ids)
            for ids, img in zip(model_inputs["chosen_input_ids"], model_inputs["images"])
        ]

        return model_inputs 

        
    def print_example(self, example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer", mm_plugin: BasePlugin = None):
        chosen_ids = example["chosen_input_ids"]
        rejected_ids = example["rejected_input_ids"]

        print("*"*50)
        print("chosen_ids:\n{}".format(chosen_ids))
        print("*"*50)
        print("rejected_ids:\n{}".format(rejected_ids))
        print("*"*50)
        print("chosen:\n{}".format(tokenizer.decode(chosen_ids, skip_special_tokens=False)))
        print("*"*50)
        print("rejected:\n{}".format(tokenizer.decode(rejected_ids, skip_special_tokens=False)))
        print("*"*50)

        mm_plugin = mm_plugin or getattr(self, "mm_plugin", None)
        if mm_plugin is not None:
            for key in ["chosen", "rejected"]:
                labels = example[f"{key}_input_ids"]
                if isinstance(labels, list):
                    import torch
                    labels = torch.tensor(labels).unsqueeze(0)  # Convert to tensor if it's a list
                    assert labels.dim() == 2, "Labels should be a 2D tensor."
                visual_mask = mm_plugin.create_visual_mask(labels, tokenizer)
                labels = labels.masked_fill(visual_mask, -100).tolist()[0]
                valid_labels = list(filter(lambda x: x != -100, labels))
                print("{}_labels:\n{}".format(key, tokenizer.decode(valid_labels, skip_special_tokens=False)))
                print("*"*50)
            