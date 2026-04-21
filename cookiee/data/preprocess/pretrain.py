from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List
import collections
from collections import Counter

from .base import BasePreprocessor
from ..mm_plugin import BasePlugin
from ...configs import DatasetArguments


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


    def old_packing(
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
        # model_inputs = {
        #     k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        #     for k, t in concatenated_examples.items()
        # }
        model_inputs = {}
        # 分别处理input_ids和attention_mask
        for k, t in concatenated_examples.items():
            model_inputs[k] = []
            for i in range(0, total_length, block_size):
                block = t[i : i + block_size]
                model_inputs[k].append(block)

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
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

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

        block_size = dataset_args.max_seq_length
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "position_ids": [], 
        }

        # --- 步骤 1: 预处理 - 切分超长样本 ---
        # 将所有样本（原始短样本 + 长样本的切片）放入一个列表中
        all_samples_to_pack = []
        for single_example_ids in tokenized_examples["input_ids"]:
            doc_len = len(single_example_ids)
            if doc_len > block_size:
                # 如果文档超长，则进行切分
                for i in range(0, doc_len, block_size):
                    # 将每个切片作为一个独立的样本添加到列表中
                    chunk = single_example_ids[i:i + block_size]
                    all_samples_to_pack.append(chunk)
            elif doc_len > 0: # 避免空样本
                # 如果文档不超长，直接添加到列表
                all_samples_to_pack.append(single_example_ids)


        # --- 步骤 2: 贪婪背包算法打包 ---
        # 按长度对所有待打包样本进行排序（从小到大）
        # 这是贪婪策略的关键，它让我们能同时访问最长和最短的样本
        sorted_samples = sorted(all_samples_to_pack, key=len)

        # 使用双端队列（deque）以便高效地从两端弹出元素
        samples_deque = collections.deque(sorted_samples)

        while samples_deque:
            # -- 开始构建一个新的 block --
            current_block_ids = []
            current_block_pos_ids = []
            # 1. 贪婪策略：首先放入当前最长的样本
            #    这能保证最难放的样本被优先处理
            longest_sample = samples_deque.pop() # 从右侧（最长）弹出一个
            current_block_ids.extend(longest_sample)
            current_block_pos_ids.extend(list(range(len(longest_sample))))
            
            # 2. 贪婪地用最短的样本填充剩余空间
            while samples_deque:
                remaining_space = block_size - len(current_block_ids)
                
                # 查看当前最短的样本是否能装下
                if len(samples_deque[0]) <= remaining_space:
                    shortest_sample = samples_deque.popleft() # 从左侧（最短）弹出一个
                    current_block_ids.extend(shortest_sample)
                    # 关键：为这个新加入的短样本生成从0开始的 position_ids
                    current_block_pos_ids.extend(list(range(len(shortest_sample))))
                else:
                    # 如果最短的都放不下，那么没有其他样本能放下了
                    break

            # -- 当前 block 填充完毕，进行 padding 并添加到最终结果中 --
            current_len = len(current_block_ids)
            #padding_len = block_size - current_len

            # a. 填充 input_ids
            final_ids = current_block_ids #+ [tokenizer.pad_token_id] * padding_len
            model_inputs["input_ids"].append(final_ids)

            # b. 创建 attention_mask
            final_mask = [1] * current_len #+ [0] * padding_len
            model_inputs["attention_mask"].append(final_mask)

            # c. 填充 position_ids
            final_pos = current_block_pos_ids #+ [0] * padding_len
            model_inputs["position_ids"].append(final_pos)

            
        # 返回最终构建好的模型输入
        return model_inputs


    def split_examples_by_domain(self, examples: Dict[str, List[Any]],):
        domains = examples["_domain"]
        examples_by_domain = {}
        for i in range(len(domains)):
            assert sum([len(value) == len(domains) for value in examples.values()]) == len(examples)
            if domains[i] not in examples_by_domain:
                examples_by_domain[domains[i]] = {}
            for key in examples:
                if key not in examples_by_domain[domains[i]]:
                    examples_by_domain[domains[i]][key] = []
                examples_by_domain[domains[i]][key].append(examples[key][i])
        return examples_by_domain


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
            if not all(examples["_domain"][0] == domain for domain in examples["_domain"]):
                model_inputs = {"input_ids": [], "attention_mask": [], "position_ids": [], "domain": []}
                examples_by_domain = self.split_examples_by_domain(examples)
                for domain, examples in examples_by_domain.items():
                    cur_packing_results = self.packing(examples, tokenizer, mm_plugin, dataset_args, chat_template, *args, **kwargs)
                    model_inputs["input_ids"].extend(cur_packing_results["input_ids"])
                    model_inputs["attention_mask"].extend(cur_packing_results[ "attention_mask"])
                    model_inputs["position_ids"].extend(cur_packing_results["position_ids"])
                    model_inputs["domain"].extend([domain] * len(cur_packing_results["input_ids"]))
            else:
                model_inputs = self.packing(examples, tokenizer, mm_plugin, dataset_args, chat_template, *args, **kwargs)
                domain = examples["_domain"][0]
                model_inputs["domain"] = [domain] * len(model_inputs["input_ids"])

            # Don't support multi-modal data in packing mode now.
            if not all(examples["_images"][i] is None for i in range(len(examples["_images"]))):
                raise ValueError("Multi-modal pretrain is not supported in packing mode now.")
            model_inputs["images"] = [None] * len(model_inputs["input_ids"])

        else:
            model_inputs = self.process(examples, tokenizer, mm_plugin, dataset_args, chat_template, *args, **kwargs)
            model_inputs["domain"] = examples['_domain']
            model_inputs["images"] = examples["_images"]

        model_inputs["length"] = [len(ids) for ids in model_inputs["input_ids"]]
        # 文本的长度为负值
        model_inputs["modality_lengths"] = [
            -len(ids) if img is None else len(ids)
            for ids, img in zip(model_inputs["input_ids"], model_inputs["images"])
        ]

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
            