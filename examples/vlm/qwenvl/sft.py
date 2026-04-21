import sys
import torch
import torch.distributed as dist
from helper import Config

from cookiee.trainer import VLMTrainer
from cookiee.configs import parse_config
from cookiee.models.utils import  freeze_layers, print_trainable_parameters, print_rank0
from cookiee.data import DatasetPipeline, collator, Qwen2vlPlugin

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor


def build_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        
    model = AutoModelForImageTextToText.from_pretrained(
        config.model, 
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.tokenizer = tokenizer

    processor = AutoProcessor.from_pretrained(config.model)

    return model, tokenizer, processor


def main(config):
    config = parse_config(config)
    # build model and tokenizer
    model, tokenizer, processor = build_model_and_tokenizer(config)
    
    # freeze vision tower
    freeze_layers(model, ["vision_tower",])
    
    for module in ["vision_tower", "multi_modal_projector", "language_model"]:
        print_rank0("*"*25 + f"{module} parameters" + "*"*25)
        print_trainable_parameters(model)
    print_rank0("*"*25 + " total parameters " + "*"*25)
    print_trainable_parameters(model)

    # build mm plugin for vlm
    mm_plugin = Qwen2vlPlugin(processor)

    # build dataset
    dataset_pipeline = DatasetPipeline(config, tokenizer, mm_plugin=mm_plugin)
    datasets = dataset_pipeline(config.task)

    # build data collator
    data_collator = collator[config.task](tokenizer=tokenizer, mm_plugin=mm_plugin)

    # build trainer
    trainer = VLMTrainer(
        model=model,
        config=config,
        processor=processor,
        image_processor=processor.image_processor,
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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/cookiee-vlm_sft.yaml"
    config = Config.fromfile(config_path)
    main(config)