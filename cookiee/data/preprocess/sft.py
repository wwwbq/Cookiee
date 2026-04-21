from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Optional

from .base import BasePreprocessor
from ..mm_plugin import BasePlugin
from ...constants import IGNORE_INDEX
from ...configs import DatasetArguments

from transformers import PreTrainedTokenizer

from helper import get_logger


logger = get_logger("sft-preprocessor")


class MultiTurnProcessor:
    """
    一个用于处理多轮对话SFT数据的高效处理器, 同时兼容单轮对话。
    它能自动检测聊天模板边界，并生成 labels。
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        start_marker: Optional[str] = None,
        end_marker: Optional[str] = None
    ):
        self.tokenizer = tokenizer

        if start_marker is None or end_marker is None:
            logger.info("Auto-detecting markers...")
            self.response_start_marker = []
            self.response_end_marker = []
            self._auto_detect_markers()
        else:
            self.response_start_marker = tokenizer.encode(start_marker, add_special_tokens=False)
            self.response_end_marker = tokenizer.encode(end_marker, add_special_tokens=False)


    def _auto_detect_markers(self):
        probe_chat = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]
        
        # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\n
        user_turn_with_gen = self.tokenizer.apply_chat_template(probe_chat[:1], tokenize=True, add_generation_prompt=True)
        
        # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nA<|im_end|>\n
        user_turn_without_gen = self.tokenizer.apply_chat_template(probe_chat[:1], tokenize=True, add_generation_prompt=False)
        
        # <|im_start|>assistant\n
        self.response_start_marker = user_turn_with_gen[len(user_turn_without_gen):]

        # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\nB<|im_end|>\n
        full_chat_ids = self.tokenizer.apply_chat_template(probe_chat, tokenize=True, add_generation_prompt=False)
        
        # <|im_start|>assistant\nB<|im_end|>\n
        assistant_part_ids = full_chat_ids[len(user_turn_without_gen):]
        
        content_b_ids = self.tokenizer.encode("B", add_special_tokens=False)

        try:
            content_start_index = assistant_part_ids.index(content_b_ids[0])
            self.response_end_marker = assistant_part_ids[content_start_index + len(content_b_ids):]
        except ValueError:
            # 没有找到end marker，尝试使用 EOS 作为 end marker
            if self.tokenizer.eos_token_id is not None:
                self.response_end_marker = [self.tokenizer.eos_token_id]
        
        # 替换 \n 方便在日志中查看
        start_marker_str = self.tokenizer.decode(self.response_start_marker).replace("\n", "\\n")
        end_marker_str = self.tokenizer.decode(self.response_end_marker).replace("\n", "\\n")

        logger.info(f"Auto-detected assistant response starts after: {start_marker_str}")
        logger.info(f"Auto-detected assistant response ends before: {end_marker_str}")


    # 查找多个子序列的所有出现位置
    def _find_all_subsequences(
        self, main_list: List[int], sub_lists: List[List[int]]
    ) -> List[Tuple[int, List[int]]]:
        """
        在主列表中查找多个子列表的所有出现位置。

        Args:
            main_list: 要在其中搜索的列表。
            sub_lists: 一个包含多个子列表的列表。

        Returns:
            一个元组列表 `(start_index, matched_sub_list)`，按 start_index 排序。
        """
        matches = []
        # 过滤掉空的子序列以避免无限循环或错误
        valid_sub_lists = [s for s in sub_lists if s]
        
        for i in range(len(main_list)):
            for sub_list in valid_sub_lists:
                if main_list[i : i + len(sub_list)] == sub_list:
                    matches.append((i, sub_list))
        
        # matches 天然按索引排序，因为外层循环是遍历 main_list
        return matches


    def _truncate_multi_turn(self, input_ids: List[int], all_markers: List[Tuple[int, List[int]]], max_seq_length: int) -> int:
        last_valid_truncation_point = -1

        # 遍历所有标记对，寻找一个完整的 [start, end] 助手回合
        for i in range(1, len(all_markers)):
            prev_pos, prev_marker = all_markers[i-1]
            curr_pos, curr_marker = all_markers[i]

            # 条件：前一个是 start 标记，当前是 end 标记
            if prev_marker == self.response_start_marker and curr_marker == self.response_end_marker:
                # 计算这个助手回合的结束位置
                end_pos = curr_pos + len(curr_marker)
                # 如果这个回合能被完整地包含在最大长度内
                if end_pos <= max_seq_length:
                    # 更新最后一个有效的截断点
                    last_valid_truncation_point = end_pos
        
        if last_valid_truncation_point != -1:
            truncation_point = last_valid_truncation_point
            final_input_ids = input_ids[:truncation_point]
            final_all_markers = [marker for marker in all_markers if marker[0] < truncation_point]
            return final_input_ids, final_all_markers
        else:
            # 没有任何一个完整的助手回复能被装下，则不截断，在后续会被直接drop
            return input_ids, all_markers


    def _truncate_one_turn(self, prompt_len: int, label_len: int, max_seq_length: int):
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


    def build_multi_turn_labels(self, input_ids: List[int], max_seq_length: int) -> Dict[str, List[int]]:
        
        # 1. 一次性找到所有 start 和 end 标记
        all_markers = self._find_all_subsequences(
            input_ids, [self.response_start_marker, self.response_end_marker]
        )
        
        # 2. 如果序列超长，则利用已找到的标记计算截断点, 对input_ids进行截断
        if len(input_ids) > max_seq_length:
            input_ids, all_markers = self._truncate_multi_turn(input_ids, all_markers, max_seq_length)
            # 如果仍然超长，说明没有一次完整的问答能被装下，直接返回，后续直接drop
            if len(input_ids) > max_seq_length:
                return input_ids, [IGNORE_INDEX] * len(input_ids)
        
        # 这里要使用截断后的input_ids来构建labels !!! 不然labels的长度和input_ids不一致
        labels = [IGNORE_INDEX] * len(input_ids)

        # 3. 遍历找到的标记，进行配对和赋值
        i = 0
        while i < len(all_markers):
            start_pos, marker = all_markers[i]
            
            # 如果当前标记是开始标记
            if marker == self.response_start_marker:
                response_content_start_pos = start_pos + len(marker)
                
                # 向前查找下一个结束标记
                next_end_pos = -1
                # 从下一个标记开始查找
                for j in range(i + 1, len(all_markers)):
                    future_pos, future_marker = all_markers[j]
                    if future_marker == self.response_end_marker:
                        next_end_pos = future_pos + len(self.response_end_marker)
                        break # 找到了最近的结束标记
                
                # 如果没找到结束标记，说明这是最后一轮对话，标记到序列末尾
                if next_end_pos == -1:
                    labels[response_content_start_pos:] = input_ids[response_content_start_pos:]
                    break # 处理完毕，退出循环
                else:
                    # 找到了配对的结束标记，标记中间的内容
                    labels[response_content_start_pos:next_end_pos] = input_ids[response_content_start_pos:next_end_pos]
            
            i += 1

        # 确保 input_ids 末尾的 EOS token (如果存在) 被学习
        if  input_ids[-1] == self.tokenizer.eos_token_id and labels[-1] == IGNORE_INDEX:
            labels[-1] = self.tokenizer.eos_token_id

        return input_ids, labels
   

    def build_one_turn_labels(self, prompt_ids, label_ids, max_seq_length, use_bos_token, use_eos_token):
        prompt_len, label_len = self._truncate_one_turn(len(prompt_ids), len(label_ids), max_seq_length)

        prompt_ids = prompt_ids[:prompt_len]
        if use_bos_token:
            prompt_ids = [self.tokenizer.bos_token_id] + prompt_ids

        label_ids = label_ids[:label_len]
        if use_eos_token:
            label_ids += [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + label_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + label_ids

        return input_ids, labels
    

    def __call__(
        self, 
        prompt_ids: List[int], 
        label_ids: List[int], 
        dataset_args: DatasetArguments,
        is_multi_turn: bool = True
    ) -> Any:
        if is_multi_turn:
            if dataset_args.use_bos_token:
                prompt_ids = [self.tokenizer.bos_token_id] + prompt_ids
            if dataset_args.use_eos_token:
                label_ids += [self.tokenizer.eos_token_id]
            input_ids = prompt_ids + label_ids
            max_seq_length = dataset_args.max_seq_length
            return self.build_multi_turn_labels(input_ids, max_seq_length)
        else:
            use_bos_token = dataset_args.use_bos_token
            use_eos_token = dataset_args.use_eos_token
            max_seq_length = dataset_args.max_seq_length
            return self.build_one_turn_labels(prompt_ids, label_ids, max_seq_length, use_bos_token, use_eos_token)


class SftPreprocessor(BasePreprocessor):
    drop_ids = []
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
                self.drop_ids.append(i)
                continue
            
            system = examples["_system"][i]
            prompt = examples["_prompt"][i]
            response = examples["_response"][i]

            is_multi_turn = len(prompt) > 1 and not dataset_args.mask_history

            if system is not None:
                if isinstance(system, str):
                    system = [{"role": "system", "content": system}]
                prompt = system + prompt
            
            if mm_plugin is not None:
                prompt = mm_plugin.preprocess_multi_modal_messages(prompt, images=examples["_images"][i])

            prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, chat_template=chat_template)
            label_ids = tokenizer(response[0]["content"], add_special_tokens=False).input_ids

            if not hasattr(self, "multi_turn_processor"):
                self.multi_turn_processor = MultiTurnProcessor(tokenizer)
            
            input_ids, labels = self.multi_turn_processor(prompt_ids, label_ids, dataset_args, is_multi_turn)

            # 如果截断后还超长，说明没有合适的截断点，则丢弃该样本
            if len(input_ids) > dataset_args.max_seq_length or len(labels) > dataset_args.max_seq_length:
                logger.warning("Dropped too long example: {}".format(examples["_prompt"][i] + examples["_response"][i]))
                self.drop_ids.append(i)
                continue

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

        self.drop_ids = []

        if dataset_args.packing:
            model_inputs = self.packing(examples, tokenizer, mm_plugin, dataset_args, chat_template, *args, **kwargs)
        else:
            model_inputs = self.process(examples, tokenizer, mm_plugin, dataset_args, chat_template, *args, **kwargs)

        keep_ids = [i for i in range(len(examples["_prompt"])) if i not in set(self.drop_ids)]

        model_inputs["images"] = [examples["_images"][i] for i in keep_ids]
        model_inputs["domain"] = [examples["_domain"][i] for i in keep_ids]

        # 文本的长度为负值
        model_inputs["modality_lengths"] = [
            -len(ids) if img is None else len(ids)
            for ids, img in zip(model_inputs["input_ids"], model_inputs["images"])
        ]

        return model_inputs 


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
