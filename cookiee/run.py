import os
import sys
import subprocess
from copy import deepcopy

from transformers.utils import is_torch_available
from helper import get_logger

logger = get_logger("cli")


def get_device_count():
    if is_torch_available():
        import torch
        return torch.cuda.device_count()
    else:
        return 0


def find_available_port() -> int:
    r"""Find an available port on the local machine."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def train(script_dir: str, args: list):
    nnodes = os.getenv("NNODES", "1")
    node_rank = os.getenv("NODE_RANK", "0")
    nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", str(find_available_port()))

    logger.info(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")
    if int(nnodes) > 1:
        logger.info(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")

    env = deepcopy(os.environ)
    # NOTE: DO NOT USE shell=True to avoid security risk
    process = subprocess.run(
        (
            "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
            "--master_addr {master_addr} --master_port {master_port} {script_dir} {args}"
        )
        .format(
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


def view(task, dataset_dir=None, image_folder=None, tokenizer_path=None):
    from helper import run_terminal_command
    try:
        import streamlit # type: ignore
    except ImportError:
        logger.error("Streamlit is not installed. Please install it to use the view feature.")
        sys.exit(1)
    from cookiee.utils.dataset_viewer import pretrain, sft, preference

    assert task in ["pretrain", "sft", "preference"]

    if task == "pretrain":
        viewer_dir = pretrain.__file__
    elif task == "sft":
        viewer_dir = sft.__file__
    elif task == "preference":
        viewer_dir = preference.__file__

    command = f"streamlit run {viewer_dir}"
    if dataset_dir:
        command += f" {dataset_dir}"
    if image_folder:
        command += f" {image_folder}"
    if tokenizer_path:
        command += f" {tokenizer_path}"

    return run_terminal_command(command)


def main():
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    if command == "train":
        script_dir = sys.argv[1]
        args = sys.argv[2:]
        train(script_dir, args)
    elif command == "view":
        task = sys.argv[1] if len(sys.argv) > 1 else None
        dataset_dir = sys.argv[2] if len(sys.argv) > 2 else None
        image_folder = sys.argv[3] if len(sys.argv) > 3 else None
        tokenizer_path = sys.argv[4] if len(sys.argv) > 4 else None
        view(task, dataset_dir, image_folder, tokenizer_path)
    else:
        logger.error(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()