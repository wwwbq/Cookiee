import os
import pandas as pd
from tqdm import tqdm
from helper import read, save
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HFDataset

path = "/root/paddlejob/share-storage/gpfs/wangbingquan/hf_hub/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
image_token = "<image>"

datasets = read(path)

saved_datasets = []
for i in tqdm(range(len(datasets))):
    conversation = datasets[i]["conversations"]
    assert len(conversation) == 2
    assert image_token in conversation[0]["value"]
    content = image_token + conversation[1]["value"]

    item_id = datasets[i]["id"]
    image = datasets[i]["image"]

    saved_datasets.append({
        "id": item_id,
        "image": image,
        "content": content,
    })

save(saved_datasets, "/root/paddlejob/share-storage/gpfs/wangbingquan/hf_hub/LLaVA-Pretrain/blip_laion_cc_sbu_558k_cleaned.json")


#dataset = load_dataset("json", data_files=path, num_proc=512)["train"]
#df = pd.read_json("/root/paddlejob/wangbingquan/hf_hub/Llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k_cleaned.json")
#dataset = HFDataset.from_pandas(df, preserve_index=False)

#num_cleaned_files = dataset.cleanup_cache_files()
#print(f"成功清理了 {num_cleaned_files} 个缓存文件。")