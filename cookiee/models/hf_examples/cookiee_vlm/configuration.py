from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class CookieeVLMConfig(PretrainedConfig):
    model_type = "cookiee-vlm"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    is_composition = True

    def __init__(
        self,
        vision_model=None,
        text_model=None,
        vision_config=None,
        text_config=None,
        stage="pretrain",
        image_token_id=32000,
        projector_hidden_act="gelu",
        multimodal_projector_bias=True,
        **kwargs,
    ):
        self.vision_model = vision_model
        self.text_model = text_model
        self.stage = stage
        
        self.image_token_id = image_token_id
        self.projector_hidden_act = projector_hidden_act

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        # elif vision_config is None:
        #     vision_config = CONFIG_MAPPING["clip_vision_model"](
        #         intermediate_size=4096,
        #         hidden_size=1024,
        #         patch_size=14,
        #         image_size=336,
        #         num_hidden_layers=24,
        #         num_attention_heads=16,
        #         vocab_size=32000,
        #         projection_dim=768,
        #     )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        # elif text_config is None:
        #     text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias

        super().__init__(**kwargs)


AutoConfig.register("cookiee-vlm", CookieeVLMConfig)

