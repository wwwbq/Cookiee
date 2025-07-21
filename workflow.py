from helper import Config
from transformers.training_args import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
import torch
from functools import partial

from trainer import BaseTrainer
from configs import parse_config
from data import DatasetPipeline, collator, Qwen2vlPlugin


config_path = "test.yaml"

def run_sft(config):
    config = parse_config(config)

    model_id = config.model

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        image_processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        mm_plugin = Qwen2vlPlugin(image_processor, config.image_token)
    except:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        image_processor = None
        mm_plugin = None

    dataset_pipeline = DatasetPipeline(config, tokenizer, mm_plugin)
    datasets = dataset_pipeline(config.task)
    data_collator = collator[config.task](tokenizer=tokenizer, mm_plugin=mm_plugin)

    # 如果有的列里面的值是str，data_collator会不知道怎么对str进行pad，然后报错
    # processed_dataset = processed_dataset.remove_columns(["instruction", "input", "output"])

    trainer = BaseTrainer(
        model=model,
        config=config,
        processing_class=tokenizer,
        **datasets,
        data_collator=data_collator,
    )

    #resume_from_checkpoint = config.resume_from_checkpoint

    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    config = Config.fromfile(config_path)
    run_sft(config)
    