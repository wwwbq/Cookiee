import sys
import torch
import torch.distributed as dist
from helper import Config


from cookiee.trainer import VLMTrainer
from cookiee.configs import parse_config
from cookiee.models.utils import add_special_tokens, freeze_layers, print_trainable_parameters, print_rank0
from cookiee.models.cookiee_vlm import CookieeVlmConfig, CookieeVlmProcessor
from cookiee.data import DatasetPipeline, DatasetArguments, collator, Qwen2vlPlugin

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor, AutoImageProcessor


def build_model_and_tokenizer(config):
    ### tokenizer ###
    tokenizer_path = config.vlm.tokenizer if hasattr(config.vlm, "tokenizer") else config.vlm.text_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if "qwen2" in tokenizer_path.lower():
        tokenizer.eos_token = "<|im_end|>"
    
    if not hasattr(config.vlm, "image_token"):
        image_token_id = tokenizer.image_token
    else:
        image_token_id = tokenizer.convert_tokens_to_ids(config.vlm.image_token)

    ### model ###
    model_config = CookieeVlmConfig(
        vision_model=config.vlm.vision_model,
        text_model=config.vlm.text_model,
        projector_type="patch_merger",
        enable_mlp=False,
        stage=config.task,
        image_token_id=image_token_id,
        projector_hidden_act="gelu",
        multimodal_projector_bias=True,
    )
    model_config._attn_implementation = "flash_attention_2"
    model_config.dtype=torch.bfloat16

    model = AutoModelForImageTextToText.from_config(model_config)
    model.tokenizer = tokenizer

    ### image processor ###
    if hasattr(config.vlm, "image_processor"):
        image_processor_path = config.vlm.image_processor
    else:
        image_processor_path = config.vlm.vision_model["weight"]

    min_pixels = getattr(config.vlm, "min_pixels", 28 * 28 * 4)
    max_pixels = getattr(config.vlm, "max_pixels", 28 * 28 * 1280)
    size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}

    # Qwen2VLImageProcessor 在smart resize时是根据size的shortest_edge和longest_edge来决定的
    # 如果只传 min_pixels 和 max_pixels ，初始化processor时，min_pixels 和 max_pixels由config中的值决定，然后size会被这两个值确定
    # 传入的 min_pixels 和 max_pixels 在后续覆盖了processor的这两个属性，但是processor的size属性并没有更改
    # 因此smart resize会时不会根据设定的 min_pixels 和 max_pixels 来决定，所以必须同时传入size和min_pixels和max_pixels，将这些属性全部覆盖
    # 概括：from_pretrained会先根据config创建一个processor，然后再根据传入的参数进行覆盖
    image_processor = AutoImageProcessor.from_pretrained(
                            image_processor_path,
                            min_pixels=min_pixels,
                            max_pixels=max_pixels,
                            size=size,
                            use_fast=True
                        )

    processor = CookieeVlmProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_token=config.vlm.image_token
    )

    return model, tokenizer, processor


def main(config):
    config = parse_config(config)
    # build model and tokenizer
    model, tokenizer, processor = build_model_and_tokenizer(config)

    if hasattr(config, "new_special_tokens"):
        model, tokenizer = add_special_tokens(model, tokenizer, config.new_special_tokens)
    
    # freeze vision tower and language model
    freeze_layers(model, ["vision_tower", "language_model"])
    #freeze_layers(model, ["language_model"])

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
    mm_plugin = Qwen2vlPlugin(processor)

    # build dataset
    dataset_pipeline = DatasetPipeline(config, tokenizer, mm_plugin=mm_plugin)
    datasets = dataset_pipeline(config.task)

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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/cookiee-vlm_pretrain.yaml"
    config = Config.fromfile(config_path)
    main(config)