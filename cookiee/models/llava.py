import os
import torch
from torch import nn
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig, AutoProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.models.auto import AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.utils import is_torchdynamo_compiling
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.loss.loss_utils import fixed_cross_entropy

from . import VISION_TOWER, PROJECTOR

from helper import get_logger

logger = get_logger("llava")

rank = int(os.environ.get("LOCAL_RANK", 0))

### CONFIG ###
class LlavaVlmConfig(PretrainedConfig):
    model_type = "llava_vlm"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    is_composition = True

    def __init__(
        self,
        vision_model={"type": None, "weight": None},
        text_model={"type": None, "weight": None},
        vision_config=None,
        text_config=None,
        projector_type="mlp",
        stage="pretrain",
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        multimodal_projector_bias=True,
        **kwargs,
    ):
        self.vision_model = vision_model
        self.text_model = text_model
        self.projector_type = projector_type
        self.stage = stage
        
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        self.text_config = text_config

        self.multimodal_projector_bias = multimodal_projector_bias

        super().__init__(**kwargs)


#### MODEL ####
@dataclass
class LlavaOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class LlavaVlmPreTrainedModel(PreTrainedModel):
    config_class = LlavaVlmConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    #_no_split_modules = ["VlmVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlavaVlmForConditionalGeneration(LlavaVlmPreTrainedModel, GenerationMixin):
    def __init__(self, config: LlavaVlmConfig):
        super().__init__(config)
        
        # pretrain stage
        if config.stage == "pretrain" or "pretrain" in config.stage:
            vision_cls, vision_weight, vision_kwargs = self.build_vision_tower(config)
            if rank == 0:
                logger.info(f"loading vision model: {vision_cls.__name__} from pretrained {vision_weight}")
            # 如果自定义模型，需要实现from_pretrained方法，并且在from_pretrained中通过vision_kwargs设置参数，并加载vision_weight权重
            self.vision_tower = vision_cls.from_pretrained(vision_weight, **vision_kwargs)
            config.vision_config = self.vision_tower.config

            if rank == 0:
                logger.info(f"loading language model: {config.text_model} from pretrained")
            self.language_model = AutoModelForCausalLM.from_pretrained(config.text_model)
            config.text_config = self.language_model.config

            config.stage = "sft"

        else:
            if config.stage != "sft":
                if rank == 0:
                    logger.warning(f"stage {config.stage} is not in [pretrain, sft], vlm will not load vision and text model`s weights")
            
            vision_cls, vision_weight, vision_kwargs = self.build_vision_tower(config)
            # 如果自定义模型，需要实现from_config
            try:
                self.vision_tower = vision_cls.from_config(config.vision_config)
            except:
                self.vision_tower = vision_cls._from_config(config.vision_config)
            self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        self.multi_modal_projector = PROJECTOR.build(config.projector_type)(config)

        self.vocab_size = config.text_config.vocab_size
        
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        # self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.post_init()

    def build_vision_tower(self, config: LlavaVlmConfig):
        if isinstance(config.vision_model, dict):
            model_type, model_weight = config.vision_model["type"], config.vision_model["weight"]
            model_cls: AutoModel = VISION_TOWER.build(model_type)
            model_kwargs = {k: v for k, v in config.vision_model.items() if k not in ["type", "weight"]}

        # 不传model type, 说明可以通过AutoModel.from_pretrained来初始化
        elif isinstance(config.vision_model, str):
            model_cls, model_weight = AutoModel, config.vision_model
            model_kwargs = {}

        return model_cls, model_weight, model_kwargs

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_vision_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ):
        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            # For default; crop CLS from each hidden state in the hidden state pool
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        return selected_image_feature

    def prepare_vision_kwargs(self, **kwargs):
        if not hasattr(self.vision_tower, "main_input_name"):
            return {}, kwargs
        
        vision_kwargs = {}
        for key in self.vision_tower.main_input_name:
            if key in kwargs:
                vision_kwargs[key] = kwargs.pop(key)
        
        return vision_kwargs, kwargs

    def merge_multi_modal_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.LongTensor,
        **kwargs
    ):
        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

        if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
            n_image_tokens = (input_ids == self.config.image_token_index).sum()
            n_image_features = image_features.shape[0] * image_features.shape[1]
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, LlavaOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            #pixel_values = pixel_values.type(self.vision_tower.get_dtype())
            vision_kwargs, kwargs = self.prepare_vision_kwargs(**kwargs)
            selected_image_feature = self.get_vision_features(pixel_values=pixel_values, **vision_kwargs)

            image_features = self.multi_modal_projector(selected_image_feature)

            inputs_embeds = self.merge_multi_modal_features(input_ids, inputs_embeds, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            num_items_in_batch = kwargs.pop("num_items_in_batch", None)
            loss = fixed_cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device),
                num_items_in_batch=num_items_in_batch, **kwargs
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


### processor ###
class LlavaVlmProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "image_token",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_seqlen=576,
        image_token="<image>", 
        **kwargs,
    ):
        self.image_seqlen = image_seqlen
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        if images is not None:
            image_inputs = self.image_processor(images, **kwargs)
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            num_image_tokens = self.image_seqlen

            prompt_strings = []
            for sample in text:
                sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
                prompt_strings.append(sample)

        text_inputs = self.tokenizer(prompt_strings, **kwargs)
        
        return BatchFeature(data={**text_inputs, **image_inputs})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


### Register to HF ###
AutoConfig.register("llava_vlm", LlavaVlmConfig)
AutoModelForImageTextToText.register(LlavaVlmConfig, LlavaVlmForConditionalGeneration)
AutoProcessor.register(LlavaVlmConfig, LlavaVlmProcessor)