import torch
from torch import nn

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig, Qwen2VLConfig

from ..utils import load_safetensors_from_directory, check_loaded_weight
from . import VISION_TOWER


class QwenVit(Qwen2VisionTransformerPretrainedModel):
    config_class = Qwen2VLVisionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False  # TODO (joao): fix. torch.compile failing probably due to `cache_positions`

    main_input_name = ["hidden_states", "image_grid_thw", "video_grid_thw"]

    def __init__(self, config) -> None:
        super().__init__(config)
        del self.merger

    def _init_weights(self, module):
        # std = self.config.initializer_range
        std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, hidden_states: torch.Tensor, image_grid_thw: torch.Tensor, video_grid_thw: torch.Tensor = None) -> torch.Tensor:
        assert not (image_grid_thw is not None and video_grid_thw is not None), "image_grid_thw and video_grid_thw can`t both exists"
        grid_thw = image_grid_thw if video_grid_thw is None else video_grid_thw
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        return hidden_states

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    #     if pretrained_model_name_or_path.split("/")[-1] != "Qwen2-VL-7B":
    #         return cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
    #     else:
    #         qwen2vl_config = Qwen2VLConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    #         vision_config = qwen2vl_config.vision_config
            
    #         weight_dict = load_safetensors_from_directory(pretrained_model_name_or_path)
    #         weight_dict = {k.replace('visual.', '') : v for k, v in weight_dict.items() if 'visual' in k and "merger" not in k}
            
    #         qwen_vit = cls(vision_config)
    #         qwen_vit.load_state_dict(weight_dict, strict=True)
    #         check_loaded_weight(qwen_vit, weight_dict)

    #         return qwen_vit


    

VISION_TOWER._do_register("qwen_vit", QwenVit)