from configs import parse_config
from data import DatasetProcessor
from helper import Config
from transformers import AutoTokenizer


config = Config.fromfile('test.yaml')
config = parse_config(config)

tokenizer = AutoTokenizer.from_pretrained(config.model)

dataset_processor = DatasetProcessor(config)

datasets = dataset_processor.build(config.task, tokenizer)
