import argparse
from helper import Config

from cookiee.configs import DatasetArguments
from cookiee.data import DatasetPipeline

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument("--data_cache_path", type=str, default=None, help="Data cache path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--task", type=str, default=None, help="Task type (e.g., pretrain, sft, preference)")
    parser.add_argument("--reader_worker", type=str|int, default="auto", help="Number of reader workers or 'auto'")

    args = parser.parse_args()

    # Load config
    config = Config.fromfile(args.config_path)

    # Override config with command line arguments (except split)
    if args.data_cache_path is not None:
        config.dataset_args.dataset_cache_path = args.data_cache_path

    if args.task is not None:
        config.task = args.task

    if args.reader_worker is not None:
        if args.reader_worker == "auto":
            config.reader_worker = "auto"
        else:
            config.reader_worker = int(args.reader_worker)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

    # Initialize dataset pipeline
    dataset_pipeline = DatasetPipeline(config, tokenizer=tokenizer)

    # Process
    dataset_pipeline(
        config.task,
        split=args.split,
        reader_worker=getattr(config, "reader_worker", 1)
    )


if __name__ == "__main__":
    main()