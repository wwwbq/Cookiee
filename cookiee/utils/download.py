import os
from modelscope.hub.snapshot_download import snapshot_download

repo_id = "Qwen/Qwen2-VL-2B-Instruct"
local_dir = os.path.join("/home/work/workspace/hf_hub", repo_id)

if not os.path.exists(local_dir):
    os.makedirs(local_dir)

snapshot_download(repo_id, local_dir=local_dir)