# modified from LlamaFactory
from itertools import chain
from io import BytesIO
from PIL import Image
from PIL.Image import Image as ImageObject
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence

import torch
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, PreTrainedTokenizer

from data import BasePlugin


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    # Create a block-diagonal mask.
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    # Use the lower triangular mask to zero out the upper triangular part
    attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d


@dataclass
class MultiModalPretrainCollator(DataCollatorForLanguageModeling):
    tokenizer: PreTrainedTokenizer = None
    mm_plugin: BasePlugin = None
    pad_to_multiple_of: Optional[int] = 8
    mlm: bool = False

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        batch_images, batch_channels, batch_lengths = [], [], []

        for feature in features:
            images = feature.pop("images", None) or []
            channels = feature.pop("channel", None) or []
            lengths = feature.pop("length", None) or []

            batch_images.extend(images)
            batch_channels.append(channels)
            batch_lengths.append(lengths)

        features: Dict[str, "torch.Tensor"] = super().__call__(features)
        
        if self.mm_plugin is not None:
            visual_inputs = self.mm_plugin.collator_fn(batch_images)
            visual_mask = self.mm_plugin.create_visual_mask(features["labels"], self.tokenizer)
            features["labels"] = features["labels"].masked_fill(visual_mask, -100)
        else:
            visual_inputs = {}

        features.update(visual_inputs)
        features["channel"] = batch_channels
        features["length"] = batch_lengths

        return features


@dataclass
class MultiModalSFTCollator(DataCollatorForSeq2Seq):
   
    tokenizer: PreTrainedTokenizer = None
    mm_plugin: BasePlugin = None
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        batch_images, batch_channels, batch_lengths = [], [], []

        for feature in features:
            images = feature.pop("images", None) or []
            channels = feature.pop("channel", None) or []
            lengths = feature.pop("length", None) or []

            batch_images.extend(images)
            batch_channels.append(channels)
            batch_lengths.append(lengths)
        
        if self.mm_plugin is not None:
            visual_inputs = self.mm_plugin.collator_fn(batch_images)
        else:
            visual_inputs = {}

        features: Dict[str, "torch.Tensor"] = super().__call__(features)

        features.update(visual_inputs)
        features["channel"] = batch_channels
        features["length"] = batch_lengths

        return features


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalSFTCollator):
    r"""
    Data collator for 4d attention mask.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        return features


collator = {
    "pretrain": MultiModalPretrainCollator,
    "sft": MultiModalSFTCollator,
}
