from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl
from helper import Config, get_logger
from dataclasses import fields
from transformers import PrinterCallback, ProgressCallback, TrainerCallback, PreTrainedModel
from transformers.integrations import is_deepspeed_zero3_enabled
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
import torch
import logging

from ..configs import parse_config
from ..loss import FocalLoss
from ..callbacks import TextLoggerCallback


logger = get_logger(__name__)


def setup_hf_logger():
    #修改hugging face的logger formatter
    formatter = logging.Formatter(
        "[%(asctime)s] - hugging face - %(levelname)s - %(message)s"
        )
    
    hf_logger = logging.getLogger("transformers")

    # 移除现有的处理器（如果有）
    for handler in hf_logger.handlers[:]:
        hf_logger.removeHandler(handler)
    
    hf_stream_handler = logging.StreamHandler()
    hf_stream_handler.setFormatter(formatter)
    hf_logger.addHandler(hf_stream_handler)


def create_peft_model(model: PreTrainedModel, config, is_train=True):
    adapter = config.peft.pop("adapter", None)
    peft_config = config.pop("peft")

    lora_config = LoraConfig(**peft_config)
    lora_config.inference_mode = not is_train
    lora_config.task_type = TaskType.CAUSAL_LM

    # if not use the main branch of transformers while occur errs
    # 看起来是加lora以后，因为只让qkv的lora模块可训练，embedding层被设为不可训练，所以输入经过embedding层后, requieres_grads变成False
    # enable_input_require_grads方法就是让经过embedding层后，requieres_grads为True
    # 也可能疑似是model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if adapter is None:  # 没有adapter时, 在get_peft_model中创建adapter
        model = get_peft_model(model, lora_config)
        logger.info(f"Created peft model successfully!")
    else: # 有adapter时, 从预训练的adapter中加载adapter
        model = PeftModel.from_pretrained(model, adapter, is_trainable=is_train)

    return model


# TODO monitor功能还没调试好
class BaseTrainer(Trainer):
    def __init__(self, config: Config, model, *args, **kwargs):
        if not isinstance(getattr(config, "training_args", {}), TrainingArguments):
            config = parse_config(config)
        
        self.config = config

        setup_hf_logger()

        if hasattr(self.config, "peft"):
            model = create_peft_model(model, config)
        
        super().__init__(args=self.config.training_args, model=model, *args, **kwargs)

        self.add_callback(TextLoggerCallback())
        for callback in [PrinterCallback, ProgressCallback]:
            self.remove_callback(callback)
        
        # for custom metric
        self.context = dict()
        if hasattr(self.config, "monitor"):
            self.context.update({metric: [] for metric in self.config.monitor.split(",")})


    def log(self, logs: dict, *args, **kwargs):
        logs.update(self.context)
        super().log(logs, *args, **kwargs)


    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        forward_kwargs = {
            "channel": inputs.pop("channel", None),
            "length": inputs.pop("length", None),
            #"num_images_per_sample": inputs.pop("num_images_per_sample", None),
        }

        if getattr(self.config, "loss", None) is None:
            loss, outputs =  super().compute_loss(model, inputs, return_outputs=True, *args, **kwargs)
        else:
            # pop出label，使得forward时不会计算loss，loss的计算就仅仅在本函数发生
            labels = inputs.pop("labels") 
            assert labels is not None 

            outputs = model(**inputs)
            
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            logits = outputs["logits"]

            forward_kwargs.update(self.config.loss)

            loss = self.loss_func(logits, labels, reduction="mean", focal_loss=True, **forward_kwargs)
            outputs["loss"] = loss
        
        for key, value in outputs.items():
            if key in self.context:
                if isinstance(value, torch.Tensor):
                    self.context[key] = value.detach().cpu().mean().item()
                else:
                    self.context[key] = value

        return (loss, outputs) if return_outputs else loss


    def loss_func(self, logits, labels, reduction="mean", focal_loss=False, **kwargs):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        if focal_loss:
            loss_fct = FocalLoss(reduction=reduction)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction) 

        try:
            vocab_size = self.model.config.vocab_size
        except:
            vocab_size = self.model.config.text_config.vocab_size

        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return loss
