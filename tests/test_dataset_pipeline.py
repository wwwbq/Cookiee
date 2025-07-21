from configs import parse_config
from helper import Config
from transformers import AutoTokenizer, AutoProcessor
from data import DatasetPipeline, Qwen2vlPlugin

config_path = "test.yaml"
config = Config.fromfile(config_path)
config = parse_config(config)

tokenizer = AutoTokenizer.from_pretrained(config.model)
image_processor = AutoProcessor.from_pretrained(config.model)
mm_plugin = Qwen2vlPlugin(image_processor, config.image_token)

dataset_pipeline = DatasetPipeline(config, tokenizer, mm_plugin)

datasets = dataset_pipeline(task=config.task, tokenizer=tokenizer)

print(1)