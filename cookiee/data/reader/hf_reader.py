import os
from typing import Dict, Optional

from datasets import load_dataset

from .base import BaseReader


class HfReader(BaseReader):
    def _build_data_files(self, dataset_path: str, dataset_spec) -> Optional[Dict[str, str]]:
        if dataset_spec.data_files is None:
            return None

        dataset_folder = dataset_spec.folder or ""
        data_files = {}
        for split, file_path in dataset_spec.data_files.items():
            if os.path.isabs(file_path) or not dataset_folder:
                data_files[split] = file_path
            else:
                data_files[split] = os.path.join(dataset_folder, file_path)

        return data_files

    def _get_hf_source(self, dataset_path: str, dataset_spec) -> str:
        if dataset_spec.subset is not None or dataset_spec.data_files is not None:
            return dataset_spec.file

        suffix = os.path.splitext(dataset_path)[1].lower()
        if suffix in [".json", ".jsonl"]:
            return "json"

        return dataset_path

    def read(self, dataset_path, dataset_spec, dataset_args):
        data_files = self._build_data_files(dataset_path, dataset_spec)
        hf_source = self._get_hf_source(dataset_path, dataset_spec)

        load_kwargs = dict(
            path=hf_source,
            name=dataset_spec.subset,
            split=dataset_spec.split,
            streaming=dataset_args.streaming,
        )
        if data_files is not None:
            load_kwargs["data_files"] = data_files
        elif hf_source == "json":
            load_kwargs["data_files"] = dataset_path

        dataset = load_dataset(**load_kwargs)

        if not dataset_args.streaming and dataset_args.max_samples is not None:
            max_samples = min(dataset_args.max_samples, len(dataset))
            dataset = dataset.select(range(max_samples))

        return dataset
