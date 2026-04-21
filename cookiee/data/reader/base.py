import os
from typing import List, Iterator

from datasets import Dataset

from helper import get_logger

from ..spec import DatasetSpec
from ...configs import DatasetArguments


logger = get_logger("reader")


class BaseReader:
    def read(self, dataset_path: str, dataset_spec: DatasetSpec, dataset_args: DatasetArguments) -> Dataset:
        raise NotImplementedError

    def _truncate(self, records: List[dict], dataset_args: DatasetArguments) -> List[dict]:
        if dataset_args.max_samples is None:
            return records

        max_samples = min(dataset_args.max_samples, len(records))
        logger.info("Sample {} from datasets ..".format(max_samples))
        return records[:max_samples]


class BaseGeneratorReader:
    def read(self, dataset_path: str, dataset_spec: DatasetSpec, dataset_args: DatasetArguments) -> Dataset:
        return Dataset.from_generator(
            self._generate_records,
            gen_kwargs={
                "dataset_path": dataset_path,
                "dataset_spec": dataset_spec,
                "dataset_args": dataset_args,
            },
            # hf的from_generator和load_datasets在多rank时会自动产生lock，保证只有一个rank在产生cache，其他rank等待完成后直接读取
            cache_dir=self._get_online_cache_path(dataset_args, "reader", dataset_spec.dataset_name),
        )

    def _get_online_cache_path(self, dataset_args: DatasetArguments, stage, dataset_name):
        if dataset_args.online_cache_path is False:
            raise ValueError("Please set `online_cache_path` when using `generator reader`")
        if dataset_args.online_cache_path is None:
            logger.warning("`online_cache_path` is not set, dataset cache of reader will store in default hugging-face`s cache dir (~/.huggingface/XXX)")
            return None
        return os.path.join(dataset_args.online_cache_path, stage, dataset_name)

    def _generate_records(
        self,
        dataset_path: str,
        dataset_spec: DatasetSpec,
        dataset_args: DatasetArguments,
    ) -> Iterator[dict]:
        max_samples = dataset_args.max_samples
        if max_samples is not None:
            logger.info("Sample {} from datasets ..".format(max_samples))

        for idx, record in enumerate(self.iter_records(dataset_path, dataset_spec, dataset_args)):
            if max_samples is not None and idx >= max_samples:
                break
            yield record

    def iter_records(
        self,
        dataset_path: str,
        dataset_spec: DatasetSpec,
        dataset_args: DatasetArguments,
    ) -> Iterator[dict]:
        raise NotImplementedError


def infer_reader_type(dataset_path: str, dataset_spec: DatasetSpec) -> str:
    reader_type = getattr(dataset_spec, "reader", None)
    if reader_type is not None:
        return reader_type

    suffix = os.path.splitext(dataset_path)[1].lower()
    if suffix == ".jsonl":
        return "jsonl"

    return "json"
