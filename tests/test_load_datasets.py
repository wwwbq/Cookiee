from datasets import load_dataset

alpaca_path = "../../Trainer/data/identity.json"
sharegpt_path = "../../Trainer/data/mllm_demo.json"

alpaca_dataset = load_dataset("json", data_files=alpaca_path)["train"]
sharegpt_dataset = load_dataset("json", data_files=sharegpt_path)["train"]

print(1)
