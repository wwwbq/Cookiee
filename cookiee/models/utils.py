import os
import math
import torch
from contextlib import nullcontext
from helper import get_logger
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled
from safetensors import safe_open


logger = get_logger("cookiee.models.utils")


def print_rank0(msg):
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        logger.info(msg)


def _noisy_mean_initialization(embed_weight: "torch.Tensor", num_new_tokens: int) -> None:
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def resize_embedding(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
    r"""Resize token embeddings."""
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [model.get_input_embeddings().weight]
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)

        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
        raise ValueError("Current model does not support resizing embedding layers.")

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    with context_maybe_zero3:
        new_embedding_size = model.get_input_embeddings().weight.size(0)
        num_new_tokens = new_embedding_size - current_embedding_size
        
        _noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)
        _noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)

    model.config.vocab_size = new_embedding_size

    return model


def add_special_tokens(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, new_tokens: list):
    if isinstance(new_tokens, str):
        new_tokens = [new_tokens]

    print_rank0(f"before adding tokens: {len(tokenizer)}")
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print_rank0(f"after adding tokens: {len(tokenizer)}")
    
    embedding_size = model.get_input_embeddings().num_embeddings
    if  embedding_size < len(tokenizer):
        model = resize_embedding(model, tokenizer)
        print_rank0(f"resized embedding layer size from {embedding_size} to {len(tokenizer)}")
    return model, tokenizer


def freeze_layers(model: PreTrainedModel, freeze_prefixes: list):
    if isinstance(freeze_prefixes, str):
        freeze_prefixes = [freeze_prefixes]

    for name, param in model.named_parameters():
        for prefix in freeze_prefixes:
            if name.startswith(prefix):
                param.requires_grad = False
                break
    
    print_rank0(f"layers with prefix: {' '.join(freeze_prefixes)} are frozen")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print_rank0(
        f"可训练参数: {trainable_params} || 总参数: {all_param} || 可训练比例: {100 * trainable_params / all_param:.2f}%"
    )


def load_safetensors_from_directory(directory_path):
    # 获取目录下所有 safetensors 文件
    safetensors_files = [f for f in os.listdir(directory_path) if f.endswith('.safetensors')]
    
    # 载入每个 safetensors 文件，并合并权重字典
    all_tensors = {}
    
    for safetensors_file in safetensors_files:
        file_path = os.path.join(directory_path, safetensors_file)
        
        # 打开 safetensors 文件并提取权重
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # 合并权重字典
                all_tensors[key] = tensor
    
    return all_tensors


def check_loaded_weight(model, weight_dict):
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        assert key in weight_dict, f"model key: {key} not in weight dict"
        assert torch.equal(model_state_dict[key], weight_dict[key]), f"model key: {key}`s weight value not equal weight dict"
    return True
