import torch
import torch.distributed as dist
import sys

from helper import Config

from cookiee.trainer import VLMTrainer
from cookiee.configs import parse_config
from cookiee.models.utils import add_special_tokens, freeze_layers, print_trainable_parameters, print_rank0
from cookiee.models.llava import LlavaVlmConfig, LlavaVlmProcessor
from cookiee.models import build_image_processor
from cookiee.data import DatasetPipeline, DatasetArguments, collator, LlavaPlugin, VICUNA_CHAT_TEMPLATE

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor, AutoImageProcessor


def build_model_and_tokenizer(config):
    ### tokenizer ###
    tokenizer_path = config.vlm.tokenizer if hasattr(config.vlm, "tokenizer") else config.vlm.text_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if "vicuna" in tokenizer_path.lower() and not hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
    if "qwen2" in tokenizer_path.lower():
        tokenizer.eos_token = "<|im_end|>"

    ### model ###
    model_config = LlavaVlmConfig(
        vision_model=config.vlm.vision_model,
        text_model=config.vlm.text_model,
        stage=config.task,
        #image_token_index=tokenizer.convert_tokens_to_ids(config.vlm.image_token),
        projector_type="mlp",
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        multimodal_projector_bias=True,
    )

    # 4.55.0(maybe)后，torch_dtyp更名为dtype, 
    # 同时在from_config时不支持传入dtype和attn_implementation，在from_pretrained时才支持
    # 或 model_config._attn_implementation = "flash_attention_2", model_config.dtype=torch.bfloat16
    model_config._attn_implementation = "flash_attention_2"
    model_config.dtype=torch.bfloat16
    
    model = AutoModelForImageTextToText.from_config(model_config)
    model.tokenizer = tokenizer

    ### image processor ###
    image_processor_configs = config.vlm.image_processor
    if isinstance(image_processor_configs, str):
        image_processor = AutoImageProcessor.from_pretrained(image_processor_configs)
    else:
        image_processor_cls, image_process_kwargs = build_image_processor(image_processor_configs.pop("type"), **image_processor_configs)
        image_processor = image_processor_cls(**image_process_kwargs)

    # llava 默认576，siglip类型的动态token数模型，会酱token数填充到max_num_patches
    image_seqlen = 576 if not hasattr(image_processor, "max_num_patches") else image_processor.max_num_patches

    processor = LlavaVlmProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_token=config.vlm.image_token,
        image_seqlen=image_seqlen,
    )

    return model, tokenizer, processor


def main(config):
    config = parse_config(config)
    # build model and tokenizer
    model, tokenizer, processor = build_model_and_tokenizer(config)

    if hasattr(config, "new_special_tokens"):
        model, tokenizer = add_special_tokens(model, tokenizer, config.new_special_tokens)
    # 初始化时tokenizer没有添加image token
    model.config.image_token_index = tokenizer.convert_tokens_to_ids(config.vlm.image_token)
    
    # freeze vision tower and language model
    freeze_layers(model, ["vision_tower", "language_model"])

    # 使用gradient checkpointing需要打开input_embeddings的grad
    if config.training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    for module in ["vision_tower", "multi_modal_projector", "language_model"]:
        print_rank0("*"*25 + f" {module} parameters " + "*"*25)
        print_trainable_parameters(getattr(model, module))
    print_rank0("*"*25 + " total parameters " + "*"*25)
    print_trainable_parameters(model)

    # build mm plugin for vlm
    mm_plugin = LlavaPlugin(processor)

    # build dataset
    dataset_pipeline = DatasetPipeline(config, tokenizer, mm_plugin=mm_plugin)
    datasets = {"train_dataset": None, "eval_dataset": None}
    datasets["train_dataset"] = dataset_pipeline(config.task, tokenizer, split="train", reader_worker=config.get("reader_worker", 1))
    if hasattr(config, "eval_dataset"):
        datasets["eval_dataset"] = dataset_pipeline(config.task, tokenizer, split="eval", reader_worker=config.get("reader_worker", 1))

    # build data collator
    data_collator = collator[config.task](tokenizer=tokenizer, mm_plugin=mm_plugin)

    # build trainer
    trainer = VLMTrainer(
        model=model,
        config=config,
        processor=processor,
        image_processor=processor.image_processor,
        processing_class=tokenizer,
        **datasets,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/vlm/llava_pretrain.yaml"
    config = Config.fromfile(config_path)
    main(config)