import os
import sys
from modelscope.hub.snapshot_download import snapshot_download as ms_download
from huggingface_hub import snapshot_download as hf_download

# repo_id = "Qwen/Qwen2-VL-2B-Instruct"
# local_dir = os.path.join("/home/work/workspace/hf_hub", repo_id)

repo_id = sys.argv[1]
local_dir = sys.argv[2]
repo_type = sys.argv[3] if len(sys.argv) > 3 else "model"
engine = sys.argv[4] if len(sys.argv) > 4 else "ms"
allow_patterns = sys.argv[5] if len(sys.argv) > 5 else None
#allow_patterns = "".join(allow_patterns.split(",")) if allow_patterns else None

local_dir = os.path.join(local_dir, repo_id)
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

if engine == "ms":
    ms_download(repo_id, local_dir=local_dir, repo_type=repo_type, allow_patterns=allow_patterns)
elif engine == "hf":
    hf_download(repo_id, local_dir=local_dir, repo_type=repo_type, allow_patterns=allow_patterns)

# python utils/download.py Qwen/Qwen2.5-1.5B /root/paddlejob/wangbingquan/hf_hub/ model ms