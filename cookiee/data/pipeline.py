import os
import sys
import json
from functools import partial
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union

from helper import get_logger
from transformers import PreTrainedTokenizer, ProcessorMixin
from datasets import DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from datasets import Dataset, IterableDataset

from .convertor import DatasetAttr, CONVERTORS
from .preprocess import PREPROCESSOR, BasePreprocessor
from .mm_plugin import BasePlugin
from ..constants import DATASET_INFO

from ..configs import DatasetArguments, TrainingArguments

logger = get_logger("dataset-pipeline")


class DatasetPipeline:
    def __init__(self, config, tokenizer: PreTrainedTokenizer = None, mm_plugin: BasePlugin = None):
        # dataset arguments
        self.args: DatasetArguments = config.dataset_args
        
        self.dataset_attr: Dict[str, DatasetAttr] = {}

        with open(os.path.join(self.args.dataset_dir, DATASET_INFO), "r") as f:
            info = json.load(f)
        
        for name in self.args.dataset.replace(" ", "").split(","):
            self.dataset_attr[name] = DatasetAttr.from_dataset_info(name, info[name])
            self.dataset_attr[name].dataset_dir = self.args.dataset_dir

        self.tokenizer = tokenizer
        self.mm_plugin = mm_plugin

        # we need some methods like ‘main_process_first’ from training arguments
        self.training_args: TrainingArguments = config.training_args


    @classmethod
    def load_from_cache(cls, tokenized_path=None, streaming=False):
        if os.path.isdir(tokenized_path) and len(os.listdir(tokenized_path)):
            logger.warning("Loading dataset from disk will ignore other data arguments.")

            dataset_dict: "DatasetDict" = load_from_disk(tokenized_path)
            
            logger.info("Loaded tokenized dataset from {}.".format(tokenized_path))

            dataset_module: Dict[str, "Dataset"] = {}
            if "train" in dataset_dict:
                dataset_module["train_dataset"] = dataset_dict["train"]
            if "validation" in dataset_dict:
                dataset_module["eval_dataset"] = dataset_dict["validation"]

            if streaming:
                dataset_module = {k: v.to_iterable_dataset() for k, v in dataset_module.items()}

            return dataset_module


    @classmethod
    def load_datasets(cls, dataset_path, dataset_args: DatasetArguments, *args, **kwargs):
        dataset = load_dataset(
            "json",
            data_files=dataset_path,
            #split=["train"],
            trust_remote_code=True,
        )["train"]

        if dataset_args.max_samples is not None:  # truncate dataset
            max_samples = min(dataset_args.max_samples, len(dataset))
            dataset = dataset.select(range(max_samples))

            if dataset_args.sample_save_path is not None:
                save_path = os.path.join(dataset_args.sample_save_path, "sample_datasets.json")
                if cls.training_args.should_save:
                    dataset.to_json(save_path, force_ascii=False) # 旧版本 ensure_ascii=False

            logger.info("Sample {} from datasets ..".format(max_samples))

        if dataset_args.streaming:  # faster than specifying streaming=True
            dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

        return dataset

    
    def convert_format(self, dataset: Dataset, dataset_attr: DatasetAttr, convertor: callable, streaming=False, preprocessing_num_workers=None):
        convert_func = partial(convertor, dataset_attr=dataset_attr)
        
        column_names = list(next(iter(dataset)).keys())

        kwargs = {}
        if not streaming:
            kwargs = dict(
                num_proc=preprocessing_num_workers,
                load_from_cache_file=False,
                desc=f"Converting format of dataset: {dataset_attr.dataset_name}",
            )
        
        return dataset.map(convert_func, batched=False, remove_columns=column_names, **kwargs)


    def merge_datasets(self, dataset_names: Sequence[str], dataset_dir: str, convertor):
        if dataset_names is None:
            return None
        
        if isinstance(dataset_names, str):
            dataset_names = dataset_names.replace(" ", "").split(",")

        datasets = []

        for name in dataset_names:
            if name not in self.dataset_attr:
                raise ValueError("dataset: [{}] not found in {}.".format(name, DATASET_INFO))

            dataset_root = self.dataset_attr[name].file_root if self.dataset_attr[name].file_root else dataset_dir
            dataset_path = os.path.join(dataset_root, self.dataset_attr[name].file_name)

            cur_datasets = self.load_datasets(dataset_path, self.args)

            converted_datasets = self.convert_format(
                                    cur_datasets, 
                                    self.dataset_attr[name], 
                                    convertor,
                                    self.args.streaming, 
                                    self.args.preprocessing_num_workers
                                )

            datasets.append(converted_datasets)
        
        if len(datasets) == 1:
            return datasets[0]
        
        return concatenate_datasets(datasets)


    def preprocess_datasets(
        self, 
        dataset: Dataset, 
        preprocessor: callable, 
        tokenizer: PreTrainedTokenizer, 
        mm_plugin: BasePlugin = None
    ):
        if dataset is None:
            return None
        
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        if mm_plugin is None:
            mm_plugin = self.mm_plugin

        preprocess_func = partial(
            preprocessor, 
            tokenizer=tokenizer, 
            mm_plugin=mm_plugin, 
            dataset_args=self.args, 
            chat_template=self.args.chat_template
        )

        column_names = list(next(iter(dataset)).keys())

        kwargs = {}
        if not self.args.streaming:
            kwargs = dict(
                num_proc=self.args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc=f"Running tokenizer on total dataset",
            )

        dataset = dataset.map(
            preprocess_func,
            batched=True,
            batch_size=self.args.preprocessing_batch_size,
            remove_columns=column_names,
            **kwargs,
        )

        return dataset


    def __call__(self, task: str, tokenizer: PreTrainedTokenizer = None):
        if self.args.tokenized_path is not None:
            return self.load_from_cache(self.args.tokenized_path)

        if tokenizer is None:
            tokenizer = self.tokenizer
            assert tokenizer is not None

        # Load and convert dataset format
        convertor = CONVERTORS[task]()
        with self.training_args.main_process_first(desc="load dataset"):
            # merge and align all datasets
            dataset = self.merge_datasets(self.args.dataset, self.args.dataset_dir, convertor)
            eval_dataset = self.merge_datasets(self.args.eval_dataset, self.args.dataset_dir, convertor)

        # preprocess dataset
        preprocessor: BasePreprocessor = PREPROCESSOR[task]()
        with self.training_args.main_process_first(desc="pre-process dataset"):
            dataset = self.preprocess_datasets(dataset, preprocessor, tokenizer)
            eval_dataset = self.preprocess_datasets(eval_dataset, preprocessor, tokenizer)

            # print dataset example
            if self.training_args.should_log:
                logger.info("training example: ")
                preprocessor.print_example(dataset[0], tokenizer, mm_plugin=self.mm_plugin)

            dataset_dict = {}
            if dataset is not None:
                if self.args.streaming:
                    dataset = dataset.shuffle(buffer_size=self.args.buffer_size, seed=self.training_args.seed)
                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if self.args.streaming:
                    eval_dataset = eval_dataset.shuffle(buffer_size=self.args.buffer_size, seed=self.training_args.seed)
                dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

            if self.args.tokenized_path is not None:
                if self.training_args.should_save:
                    dataset_dict.save_to_disk(self.args.tokenized_path)
                    logger.info("Tokenized dataset saved at {}.".format(self.args.tokenized_path))
                    logger.info("Please restart the training with `tokenized_path: {}`.".format(self.args.tokenized_path))

                sys.exit(0)

            dataset_module = {}
            if "train" in dataset_dict:
                dataset_module["train_dataset"] = dataset_dict["train"]
            if "validation" in dataset_dict:
                dataset_module["eval_dataset"] = dataset_dict["validation"]

            return dataset_module

