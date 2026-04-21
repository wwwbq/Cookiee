import os
import subprocess
import sys
from copy import deepcopy

from helper import get_logger

logger = get_logger("cli")


def get_device_count():
    import torch
    from transformers.utils import is_torch_cuda_available
    return torch.cuda.device_count() if is_torch_cuda_available() else 0


def find_available_port() -> int:
    r"""Find an available port on the local machine."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def train(script_dir, args: list):
    nnodes = os.getenv("NNODES", "1")
    node_rank = os.getenv("NODE_RANK", "0")
    nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", str(find_available_port()))

    if node_rank == 0:
        logger.info(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")

    torchrun_template = (
            "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
            "--master_addr {master_addr} --master_port {master_port} {script_dir} {args}"
        )

    env = deepcopy(os.environ)

    process = subprocess.run(
                torchrun_template.format(
                    nnodes=nnodes,
                    node_rank=node_rank,
                    nproc_per_node=nproc_per_node,
                    master_addr=master_addr,
                    master_port=master_port,
                    script_dir=script_dir,
                    args=" ".join(args),
                ).split(),
                env=env,
                check=True,
            )

    sys.exit(process.returncode)


def view(task, dataset_dir=None, image_floder=None, tokenizer_path=None):
    try: 
        import streamlit
    except:
        raise ValueError("to use dataset viwer, please run 'pip install streamlit' ")
    
    from helper import run_terminal_command
    
    from cookiee.utils.dataset_viewer import pretrain, sft, preference

    if task == "pretrain":
        viewer = pretrain.__file__
    elif task == "sft":
        viewer = sft.__file__
    elif task == "preference":
        viewer = preference.__file__

    print(viewer)

    command = f"streamlit run {viewer}"

    if dataset_dir is not None:
        command += f" {dataset_dir}"
    if image_floder is not None:
        command += f" {image_floder}"
    if tokenizer_path is not None:
        command += f" {tokenizer_path}"

    return run_terminal_command(command)

    
def main():
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    if command == "train":
        script_dir = sys.argv.pop(1)
        return train(script_dir, sys.argv[1:])
    elif command == "view":
        task = sys.argv[1]
        dataset_dir = sys.argv[2] if len(sys.argv) > 2 else None
        image_folder = sys.argv[3] if len(sys.argv) > 3 else None
        tokenizer_path = sys.argv[4] if len(sys.argv) > 4 else None
        return view(task, dataset_dir, image_folder, tokenizer_path)
    else:
        logger.info("un supprted command")
        

if __name__ == "__main__":
    main()