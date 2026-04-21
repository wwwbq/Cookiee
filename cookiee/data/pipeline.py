import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

from datasets import Dataset, concatenate_datasets, interleave_datasets, load_from_disk
from helper import get_logger
from transformers import PreTrainedTokenizer

from .convertor import CONVERTORS
from .mm_plugin import BasePlugin
from .preprocess import PREPROCESSOR, BasePreprocessor
from .reader import READERS, infer_reader_type, BaseReader
from .spec import DatasetSpec
from ..configs import DatasetArguments, TrainingArguments
from ..constants import DATASET_INFO


logger = get_logger("dataset-pipeline")


class DatasetPipeline:
    def __init__(self, config, tokenizer: PreTrainedTokenizer = None, mm_plugin: BasePlugin = None):
        self.config = config
        self.args: DatasetArguments = config.dataset_args
        self.training_args: TrainingArguments = config.training_args

        self.tokenizer = tokenizer
        self.mm_plugin = mm_plugin
        self.mix_stage = getattr(config, "mix_stage", "processed")

        if self.mix_stage not in {"converted", "processed"}:
            raise ValueError("mix_stage must be one of ['converted', 'processed']")

        self.dataset_spec = self._register_dataspec()


    @property
    def offline_cache_path(self) -> Optional[str]:
        return getattr(self.args, "offline_cache_path", None) or getattr(self.args, "tokenized_path", None)

    @property
    def online_cache_path(self) -> Optional[str]:
        return getattr(self.args, "online_cache_path", None)


    def _normalize_split(self, split: str) -> str:
        if split == "train":
            return "train"
        if split in ["validation", "val", "eval"]:
            return "validation"
        raise ValueError("Unsupported split: {}".format(split))


    def _get_split_cache_path(self, split: str) -> Optional[str]:
        if self.offline_cache_path is None:
            return None
        return os.path.join(self.offline_cache_path, self._normalize_split(split))


    def _get_online_cache_path(self, stage, dataset_name):
        if self.online_cache_path == False:
            return None
        if self.online_cache_path is None:
            return os.path.join(self.training_args.output_dir, "online_cache", stage, f"{dataset_name}.arrow")
        return os.path.join(self.online_cache_path, stage, f"{dataset_name}.arrow")


    def _parse_dataset_name(self, dataset_names: List | str | None) -> List[str]:
        if dataset_names is None:
            return []
        if isinstance(dataset_names, str):
            return [name for name in dataset_names.replace(" ", "").split(",") if name]
        if isinstance(dataset_names, list):
            return [name for name in dataset_names if name]
        raise ValueError("dataset_names must be a string or a list of strings")


    def _register_dataspec(self):
        dataset_spec: Dict[str, DatasetSpec] = {}

        with open(os.path.join(self.args.dataset_dir, DATASET_INFO), "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

        dataset_names = self._parse_dataset_name(self.args.dataset)
        eval_dataset_names = self._parse_dataset_name(self.args.eval_dataset)
        for name in dataset_names + eval_dataset_names:
            if name and name not in dataset_spec:
                dataset_spec[name] = DatasetSpec.from_dataset_info(name, dataset_info[name])

        return dataset_spec


    def _read_offline_cache(self, cache_path=None, split: str = "train", streaming: bool = False):
        if os.path.isdir(cache_path) and len(os.listdir(cache_path)):
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset = load_from_disk(cache_path)
            logger.info("Loaded cached dataset from {}.".format(cache_path))

            if streaming:
                dataset = dataset.to_iterable_dataset()

            return dataset


    def _save_offline_cache(self, dataset: Dataset, split: str = "train"):
        cache_path = self._get_split_cache_path(split)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        dataset.save_to_disk(cache_path)

        logger.info("Cached dataset saved at {}.".format(cache_path))
        logger.info("Please restart the training with `dataset_cache_path: {}`.".format(self.dataset_cache_path))
        sys.exit(0)


    def _get_dataset_names_by_split(self, split: str) -> List[str]:
        if split == "train":
            return self._parse_dataset_name(self.args.dataset)
        if split in ["validation", "val", "eval"]:
            return self._parse_dataset_name(self.args.eval_dataset)
        raise ValueError("Unsupported split: {}".format(split))


    def _get_dataset_path(self, dataset_spec: DatasetSpec):
        dataset_folder = dataset_spec.folder or ""
        return os.path.join(dataset_folder, dataset_spec.file)


    def _read(self, name: str, streaming: bool = False) -> Dataset:
        dataset_spec = self.dataset_spec[name]
        dataset_path = self._get_dataset_path(dataset_spec)

        reader_type = infer_reader_type(dataset_path, dataset_spec)
        if streaming and reader_type != "hf":
            raise ValueError("streaming mode only supports hf reader, but got {} for dataset {}.".format(reader_type, name))

        reader: BaseReader = READERS[reader_type]()
        dataset = reader.read(dataset_path, dataset_spec, self.args)

        if not streaming and len(dataset) == 0:
            raise ValueError("Empty dataset found in read stage: {}.".format(name))

        if streaming and reader_type != "hf":
            dataset = dataset.to_iterable_dataset()

        return dataset


    def read(self, dataset_names: List[str], streaming=False, reader_worker: Union[int, str] = 1):
        if not dataset_names:
            raise ValueError("No valid dataset found.")

        should_log = self.training_args.should_log
        dataset_infos = []
        sample_count_placeholder = float("inf") if streaming else None

        if streaming and should_log:
            logger.warning("Streaming dataset does not support exact sample counting; reader logs will use 'inf' as sample count.")

        if reader_worker == "auto":
            reader_worker = len(dataset_names)
        elif not isinstance(reader_worker, int):
            raise ValueError("reader_worker must be an integer or 'auto'")

        if reader_worker <= 0:
            raise ValueError("reader_worker must be greater than 0")

        if reader_worker == 1:
            datasets = []
            for name in dataset_names:
                if should_log:
                    logger.info("Reading dataset: {}.".format(name))

                dataset = self._read(name, streaming=streaming)
                sample_count = sample_count_placeholder if streaming else len(dataset)
                datasets.append(dataset)
                dataset_infos.append((name, sample_count))
        else:
            max_workers = min(reader_worker, len(dataset_names))
            if should_log:
                logger.info("Reading {} datasets with {} workers.".format(len(dataset_names), max_workers))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_info = {
                    executor.submit(self._read, name, streaming): (idx, name)
                    for idx, name in enumerate(dataset_names)
                }
                datasets = [None] * len(dataset_names)
                dataset_infos = [None] * len(dataset_names)

                for future in as_completed(future_to_info):
                    idx, name = future_to_info[future]
                    dataset = future.result()
                    sample_count = sample_count_placeholder if streaming else len(dataset)
                    datasets[idx] = dataset
                    dataset_infos[idx] = (name, sample_count)

                    if should_log:
                        logger.info("Finished reading dataset: {}. Total samples: {}.".format(name, sample_count))

        if not datasets:
            raise ValueError("No valid dataset found.")

        if should_log:
            total_samples = sum(sample_count for _, sample_count in dataset_infos)
            logger.info(
                "Finished reading datasets: {}. Total samples: {}.".format(
                    ", ".join("{}({})".format(name, sample_count) for name, sample_count in dataset_infos),
                    total_samples,
                )
            )

        return datasets


    def convert(self, datasets: List[Dataset], dataset_names: List[str], convertor):
        converted_datasets = []

        for name, dataset in zip(dataset_names, datasets):
            dataset_spec = self.dataset_spec[name]
            convert_func = partial(convertor, dataset_spec=dataset_spec)
            column_names = list(next(iter(dataset)).keys())

            kwargs = {}
            if not self.args.streaming:
                if self.args.is_shared_file_system:
                    load_from_cache_file = (not self.args.overwrite_cache) or (self.training_args.process_index != 0)
                else:
                    load_from_cache_file = (not self.args.overwrite_cache) or (self.training_args.local_process_index != 0)
                kwargs = dict(
                    num_proc=self.args.preprocessing_num_workers,
                    cache_file_name=self._get_online_cache_path("convert", name),
                    load_from_cache_file=load_from_cache_file,
                    desc=f"Convert {dataset_spec.dataset_name} ",
                )

            converted_datasets.append(
                dataset.map(convert_func, batched=False, remove_columns=column_names, **kwargs)
            )

        return converted_datasets


    def preprocess(self, datasets: List[Dataset], tokenizer: PreTrainedTokenizer, preprocessor: BasePreprocessor):
        preprocess_func = partial(
            preprocessor,
            tokenizer=tokenizer,
            mm_plugin=self.mm_plugin,
            dataset_args=self.args,
            chat_template=self.args.chat_template,
        )

        processed_datasets = []
        for idx, dataset in enumerate(datasets):
            column_names = list(next(iter(dataset)).keys())

            kwargs = {}
            if not self.args.streaming:
                dataset_name = list(self.dataset_spec.keys())[idx] if self.mix_stage != "converted" else "mixed_dataset"
                if self.args.is_shared_file_system:
                    load_from_cache_file = (not self.args.overwrite_cache) or (self.training_args.process_index != 0)
                else:
                    load_from_cache_file = (not self.args.overwrite_cache) or (self.training_args.local_process_index != 0)
                kwargs = dict(
                    num_proc=self.args.preprocessing_num_workers,
                    cache_file_name=self._get_online_cache_path("preprocess", dataset_name),
                    load_from_cache_file=load_from_cache_file,
                    desc=f"Running tokenizer on {dataset_name}",
                )

            processed_datasets.append(
                dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=self.args.preprocessing_batch_size,
                    remove_columns=column_names,
                    **kwargs,
                )
            )

        return processed_datasets


    def mix(self, datasets: List[Dataset]):
        if not datasets:
            return None
        
        if len(datasets) == 1:
            return datasets[0]

        if self.args.probabilities or self.args.streaming:
            assert self.args.probabilities and self.args.stopping_strategy, "probabilities and stopping_strategy must be provided for interleaving datasets."
            return interleave_datasets(
                datasets,
                probabilities=self.args.probabilities,
                stopping_strategy=self.args.stopping_strategy,
            )

        return concatenate_datasets(datasets)


    def __call__(self, task: str, tokenizer: PreTrainedTokenizer = None, split: str = "train", reader_worker: Union[int, str] = 1):
        if tokenizer is None:
            tokenizer = self.tokenizer
            assert tokenizer is not None

        if self.args.streaming and self.offline_cache_path is not None:
            raise ValueError("streaming mode does not support offline_cache_path")

        ### check offline cache ###
        offline_cache_path = self._get_split_cache_path(split)
        if offline_cache_path is not None and os.path.exists(offline_cache_path):
            cached_dataset = self._read_offline_cache(offline_cache_path, split=split, streaming=self.args.streaming)
            if cached_dataset is not None:
                return cached_dataset

        ### read ###
        dataset_names = self._get_dataset_names_by_split(split)
        datasets = self.read(dataset_names, streaming=self.args.streaming, reader_worker=reader_worker)

        ### convert ###
        convertor = CONVERTORS[task]()
        with self.training_args.main_process_first(desc="convert datasets", local=(not self.args.is_shared_file_system)):
            converted_datasets = self.convert(datasets, dataset_names, convertor)

        ### preprocess & mix ###
        preprocessor: BasePreprocessor = PREPROCESSOR[task]()
        with self.training_args.main_process_first(desc="preprocess datasets", local=(not self.args.is_shared_file_system)):
            if self.mix_stage == "converted":
                mixed_dataset = self.mix(converted_datasets)
                dataset = self.preprocess([mixed_dataset], tokenizer, preprocessor)[0] if mixed_dataset is not None else None
            else:
                processed_datasets = self.preprocess(converted_datasets, tokenizer, preprocessor)
                dataset = self.mix(processed_datasets)

        ### log example ###
        if dataset is not None and self.training_args.should_log:
            logger.info("{} example: ".format(split))
            preprocessor.print_example(next(iter(dataset)), tokenizer, mm_plugin=self.mm_plugin)

        ### shuffle if streaming ###
        if dataset is not None and self.args.streaming:
            dataset = dataset.shuffle(
                buffer_size=self.args.preprocessing_batch_size,
                seed=self.training_args.seed,
            )

        ### save cache ###
        if self.offline_cache_path is not None and self.training_args.should_save:
            self._save_offline_cache(dataset, split=split)

        return dataset
