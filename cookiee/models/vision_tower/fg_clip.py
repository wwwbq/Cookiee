import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, List

from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionConfig, Siglip2VisionModel, BaseModelOutputWithPooling

from . import VISION_TOWER


class Fgclip2VisionModel(Siglip2VisionModel):
    main_input_name = ["pixel_values", "pixel_attention_mask", "spatial_shapes"]

    def __init__(self, config: Siglip2VisionConfig):
        # override, we don`t need pooled out
        config.vision_use_head = False
        super().__init__(config)

        
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.
        """
        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
    

VISION_TOWER._do_register("fgclip2", Fgclip2VisionModel)