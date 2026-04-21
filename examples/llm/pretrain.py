import sys

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from cookiee import BaseTrainer, DatasetPipeline, collator, parse_config
from helper import Config


def build_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    tokenizer.pad_token = "<|fim_pad|>"

    # model = AutoModelForCausalLM.from_pretrained(
    #     config.model,
    #     # torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2"
    # )

    model_config = AutoConfig.from_pretrained(config.model)
    # 4.55.0(maybe)后，torch_dtyp更名为dtype,
    # 同时在from_config时不支持传入dtype和attn_implementation，在from_pretrained时才支持
    # 或 model_config._attn_implementation = "flash_attention_2", model_config.dtype=torch.bfloat16
    model_config._attn_implementation = "flash_attention_2"
    model_config.dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_config(model_config)

    return model, tokenizer


def main(config):
    config = parse_config(config)

    # build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(config)

    # build dataset
    dataset_pipeline = DatasetPipeline(config, tokenizer, mm_plugin=None)
    datasets = {"train_dataset": None, "eval_dataset": None}
    datasets["train_dataset"] = dataset_pipeline(config.task, tokenizer, split="train", reader_worker=config.get("reader_worker", 1))
    if hasattr(config, "eval_dataset"):
        datasets["eval_dataset"] = dataset_pipeline(config.task, tokenizer, split="eval", reader_worker=config.get("reader_worker", 1))

    # build data collator
    data_collator = collator[config.task](tokenizer=tokenizer, mm_plugin=None)

    # build trainer
    trainer = BaseTrainer(
        model=model,
        config=config,
        processing_class=tokenizer,
        **datasets,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "examples/llm/configs/pretrain.yaml"
    config = Config.fromfile(config_path)
    main(config)