import torch
import logging
from collections import defaultdict
from dataclasses import fields

from transformers import PrinterCallback, ProgressCallback, TrainerCallback, PreTrainedModel
from transformers.integrations import is_deepspeed_zero3_enabled
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl
from helper import Config, get_logger

from ..utils.average_meter import AverageMeter
from ..configs import parse_config
from ..loss import LOSSES, dispatch_loss_kwargs, MetricAggregator, BaseLoss, LossOutput
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

        # if hasattr(self.config, "domain_mapping"):
        #     self.domain_mapping = self.config.domain_mapping
        #     for domain_name in self.domain_mapping.values():
        #         self.context[f"{domain_name}_loss"] = AverageMeter(f"{domain_name}_loss")

        if self.accelerator.is_main_process:
            from copy import deepcopy
            cfg_copy: Config = deepcopy(self.config)
            cfg_copy.pop("training_args")
            cfg_copy.pop("dataset_args")
            print(cfg_copy.pretty_text)

            
    def log(self, logs: dict, *args, **kwargs):
        custom_dict = {}
        for k, v in self.context.items():
            custom_dict[k] = v.average if isinstance(v, AverageMeter) else v
        logs.update(custom_dict)

        super().log(logs, *args, **kwargs)


    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        loss_kwargs = {
            "domains": inputs.pop("domain", None),
            "length": inputs.pop("length", None),
            #"num_images_per_sample": inputs.pop("num_images_per_sample", None),
        }

        if getattr(self.config, "loss", None) is None:
            loss, outputs =  super().compute_loss(model, inputs, return_outputs=True, *args, **kwargs)
            return (loss, outputs) if return_outputs else loss
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

            loss_type = self.config.loss.type
            loss_kwargs.update(self.config.loss)
            if self.model_accepts_loss_kwargs:
                loss_kwargs.update({"num_items_in_batch": kwargs.get("num_items_in_batch", None)})

            loss = self.run_loss(
                logits, 
                labels, 
                loss_type=loss_type, 
                **loss_kwargs
            )
            outputs["loss"] = loss
        
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and kwargs.get("num_items_in_batch", None) is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


    def run_loss(
            self,
            logits,
            labels,
            loss_type,
            **loss_kwargs
    ):
        if loss_type not in LOSSES:
            raise KeyError(f"Unsupported new loss type: {loss_type}")

        loss_cls: BaseLoss = LOSSES[loss_type]
        init_kwargs, forward_kwargs, unused_kwargs = dispatch_loss_kwargs(loss_cls,**loss_kwargs,)
        loss_func = loss_cls(**init_kwargs)

        output: LossOutput = loss_func(logits, labels, **forward_kwargs)

        # BaseLoss 返回普通 tensor，BaseMetricLoss 返回带 loss 字段的对象。
        if torch.is_tensor(output):
            loss = output
        elif hasattr(output, "loss"):
            loss = output.loss
        else:
            raise TypeError(f"Unsupported loss output type: {type(output)!r}")

        if not torch.is_tensor(output) and (
            getattr(output, "scalar_metrics", None) is not None
            or getattr(output, "grouped_metrics", None) is not None
        ):
            metric_aggregator = MetricAggregator()
            merged_metrics = metric_aggregator(self.accelerator, output)
            merged_metrics = metric_aggregator.scale_metrics(merged_metrics, loss_func, self.accelerator)

            if self.accelerator.is_main_process:
                for key, value in merged_metrics.items():
                    if key not in self.context or not isinstance(self.context[key], AverageMeter):
                        self.context[key] = AverageMeter(key)
                    self.context[key].update(value)

        return loss
