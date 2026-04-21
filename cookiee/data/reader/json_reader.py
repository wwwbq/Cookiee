import json

from datasets import Dataset

from .base import BaseReader, BaseGeneratorReader


class JsonReader(BaseReader):
    def read(self, dataset_path, dataset_spec, dataset_args) -> Dataset:
        with open(dataset_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            if "train" in payload and isinstance(payload["train"], list):
                records = payload["train"]
            else:
                records = [payload]
        else:
            raise ValueError("Unsupported json payload type: {}".format(type(payload)))

        records = self._truncate(records, dataset_args)
        return Dataset.from_list(records)


class JsonlReader(BaseReader):
    def read(self, dataset_path, dataset_spec, dataset_args) -> Dataset:
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        records = self._truncate(records, dataset_args)
        return Dataset.from_list(records)


class JsonlGeneratorReader(BaseGeneratorReader):
    def iter_records(self, dataset_path, dataset_spec, dataset_args):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
