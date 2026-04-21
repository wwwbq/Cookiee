import sys
import torch
import os
import torch.distributed as dist
from helper import Config
from cookiee import DatasetPipeline, BaseTrainer, parse_config, collator
from cookiee.models.utils import add_special_tokens
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    #tokenizer.pad_token = "<|fim_pad|>"
    model = AutoModelForCausalLM.from_pretrained(
        config.model, 
        dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )
    return model, tokenizer


def main(config):
    config = parse_config(config)

    # build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(config)
    if hasattr(config, "think"):
        model, tokenizer = add_special_tokens(model, tokenizer, ["<think>", "</think>"])

    # build dataset
    dataset_pipeline = DatasetPipeline(config, tokenizer, mm_plugin=None)
    datasets = {"train_dataset": None, "eval_dataset": None}
    datasets["train_dataset"] = dataset_pipeline(config.task, tokenizer, split="train")
    if hasattr(config, "eval_dataset"):
        datasets["eval_dataset"] = dataset_pipeline(config.task, tokenizer, split="eval")

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
    
if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else "examples/llm/configs/sft.yaml"
    config = Config.fromfile(config_path)
    main(config)