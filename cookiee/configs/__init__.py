from .training_args import *
from .dataset_args import *

from helper import Config
from dataclasses import fields


def parse_config(config: Config):
    if not hasattr(config, "task"):
        raise ValueError(f"config {config.filename} needs a task field, eg: pretrain, sft")

    # 解析training_args
    if not hasattr(config, 'training_args'):
        training_fields = [field.name for field in fields(TrainingArguments)]
        training_args = {field: getattr(config, field) for field in config if field in training_fields}
        training_args = TrainingArguments(**training_args)
    else:
        training_args = config.training_args
        if isinstance(training_args, dict):
            training_args = TrainingArguments(**training_args)
        if not isinstance(training_args, TrainingArguments):
            raise ValueError(f"cannot parse training_args from config: {config.filename}")
    config.training_args = training_args

    # 解析dataset_arg
    if not hasattr(config, 'dataset_args'):
        dataset_fields = [field.name for field in fields(DatasetArguments)]
        dataset_args = {field: getattr(config, field) for field in config if field in dataset_fields}
        dataset_args = DatasetArguments(**dataset_args)
    else:
        dataset_args = config.dataset_args
        if isinstance(dataset_args, dict):
            dataset_args = DatasetArguments(**dataset_args)
        if not isinstance(dataset_args, DatasetArguments):
            raise ValueError(f"cannot parse dataset_args from config: {config.filename}")
    config.dataset_args = dataset_args

    return config
