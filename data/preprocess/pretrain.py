from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List

from .base import BasePreprocessor
from ..mm_plugin import BasePlugin
from configs import DatasetArguments

from transformers import PreTrainedTokenizer

from helper import get_logger


logger = get_logger("pretrain-preprocessor")


class PretrainPreprocessor(BasePreprocessor):
    def process(
        self,
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizer,
        mm_plugin: BasePlugin = None,
        dataset_args: DatasetArguments = None,
        chat_template=None,
    ):
        #TODO 如果不存在bos_token且use bos token，加载tokenizer时会添加默认的bos token，因此此处tokenizer一定有bos token
        bos_token = tokenizer.bos_token if dataset_args.use_bos_token else ""
        eos_token = tokenizer.eos_token if dataset_args.use_eos_token else ""

        # vlm: <image> + label

        # examples由_prompt、_response等字段组成，每个字段sample的数量由preprocessing_batch_size决定
        text_examples = [bos_token + messages + eos_token for messages in examples["_prompt"]]

        if mm_plugin is not None:
            text_examples = [
                mm_plugin.preprocess_multi_modal_messages(
                    messages=text_examples[i],
                    images=examples["_images"][i],
                )
                for i in range(len(text_examples))
            ]
        
        # include input_ids and attention_mask, attention_mask is always 1
        model_inputs = tokenizer(
            text_examples, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=dataset_args.max_seq_length
        )

        return model_inputs


    def packing(
        self,
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizer,
        mm_plugin: BasePlugin = None,
        dataset_args: DatasetArguments = None,
        chat_template=None,
    ):
        #TODO 如果不存在bos_token且use bos token，加载tokenizer时会添加默认的bos token，因此此处tokenizer一定有bos token
        bos_token = tokenizer.bos_token if dataset_args.use_bos_token else ""
        eos_token = tokenizer.eos_token if dataset_args.use_eos_token else ""

        # vlm: <image> + label

        # examples由_prompt、_response等字段组成，每个字段sample的数量由preprocessing_batch_size决定
        text_examples = [bos_token + messages + eos_token for messages in examples["_prompt"]]

        if mm_plugin is not None:
            text_examples = [
                mm_plugin.preprocess_multi_modal_messages(
                    messages=text_examples[i],
                    images=examples["_images"][i],
                )
                for i in range(len(text_examples))
            ]

         # include input_ids and attention_mask, attention_mask is always 1
        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}

        block_size = dataset_args.max_seq_length
        total_length = (len(concatenated_examples["input_ids"]) // block_size) * block_size

        # TODO: 调整截断策略，当前可能会把每个block的最后一个样本的eos token截断，导致当前block的最后一个样本没学会eos，下一个block开头就是eos
        model_inputs = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

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

            # add channel to model input
            if not all(examples["_channel"][0] == channel for channel in examples["_channel"]):
                raise ValueError("All data samples must have the same channel when packing.")
            model_inputs["channel"] = [examples["_channel"][0]] * len(model_inputs["input_ids"])

            # Don't support multi-modal data in packing mode now.
            if not all(examples["_images"][i] is None for i in range(len(examples["_images"]))):
                raise ValueError("Multi-modal pretrain is not supported in packing mode now.")
            model_inputs["images"] = [None] * len(model_inputs["input_ids"])

        else:
            model_inputs = self.process(examples, tokenizer, mm_plugin, dataset_args, chat_template, *args, **kwargs)
            model_inputs["channel"] = examples['_channel']
            model_inputs["images"] = examples["_images"]

        model_inputs["length"] = [len(ids) for ids in model_inputs["input_ids"]]

        return model_inputs 

        
    def print_example(self, example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer", mm_plugin: BasePlugin = None):
        print("*"*50)
        print("input_ids:\n{}".format(example["input_ids"]))
        print("*"*50)
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("*"*50)

        mm_plugin = mm_plugin or getattr(self, "mm_plugin", None)
        if mm_plugin is not None:
            labels = example["input_ids"]
            if isinstance(labels, list):
                import torch
                labels = torch.tensor(labels).unsqueeze(0)  # Convert to tensor if it's a list
                assert labels.dim() == 2, "Labels should be a 2D tensor."
            visual_mask = mm_plugin.create_visual_mask(labels, tokenizer)
            labels = labels.masked_fill(visual_mask, -100).tolist()[0]
            valid_labels = list(filter(lambda x: x != -100, labels))
            print("label_ids:\n{}".format(labels))
            print("*"*50)
            print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))
            