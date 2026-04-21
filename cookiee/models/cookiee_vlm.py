import os
import torch
from torch import nn
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig, AutoProcessor
from transformers.models.auto import AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.utils import is_torchdynamo_compiling
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from .llava import LlavaVlmForConditionalGeneration

from helper import get_logger

logger = get_logger("cookiee-vlm")

rank = int(os.environ.get("LOCAL_RANK", 0))

### CONFIG ###
class CookieeVlmConfig(PretrainedConfig):
    model_type = "cookiee_vlm"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    is_composition = True

    def __init__(
        self,
        vision_model={"type": None, "weight": None},
        text_model={"type": None, "weight": None},
        vision_config=None,
        text_config=None,
        projector_type="mlp",
        enable_mlp=False,
        projector_hidden_act="gelu",
        multimodal_projector_bias=True,
        stage="pretrain",
        image_token_index=151655,
        **kwargs,
    ):
        self.vision_model = vision_model
        self.text_model = text_model
        self.projector_type = projector_type
        self.enable_mlp = enable_mlp
        self.stage = stage
        
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

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
class CookieeVlmForConditionalGeneration(LlavaVlmForConditionalGeneration):
    # override settings in LlavaVlmPreTrainedModel
    config_class = CookieeVlmConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def get_vision_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_tower(pixel_values, **kwargs)

        return image_outputs

    def merge_multi_modal_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.LongTensor,
        **kwargs
    ):
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_features.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_mask = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        return inputs_embeds

    # 在继承的GenerationMixin中被调用
    # 此方法将processor处理过的输入整理成模型forward所需要的输入
    # 同时generate时会检查参数，如果传给processor的参数中有没有被此函数用到的，会报错
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        image_grid_thw=None,
        video_grid_thw=None,
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
            model_inputs["image_grid_thw"] = image_grid_thw
            model_inputs["video_grid_thw"] = video_grid_thw

        return model_inputs

### processor ###
class CookieeVlmProcessor(ProcessorMixin):
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
        image_token="<|image_pad|>", 
        **kwargs,
    ):
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
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        text_inputs = self.tokenizer(text, **kwargs)

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
AutoConfig.register("cookiee_vlm", CookieeVlmConfig)
AutoModelForImageTextToText.register(CookieeVlmConfig, CookieeVlmForConditionalGeneration)
AutoProcessor.register(CookieeVlmConfig, CookieeVlmProcessor)