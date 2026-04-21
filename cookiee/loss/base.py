import inspect
import torch
from torch import nn

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

from ..constants import IGNORE_INDEX


@dataclass
class LossOutput:
    loss: torch.Tensor
    scalar_metrics: Optional[Dict[str, torch.Tensor]] = None
    grouped_metrics: Optional[Dict[str, Dict]] = None

    def to_gather_object(self) -> list[Dict[str, Dict]]:
        return [{
            "scalar_metrics": self.scalar_metrics,
            "grouped_metrics": self.grouped_metrics,
        }]


def dispatch_loss_kwargs(loss_cls: nn.Module, **kwargs):
    init_signature = inspect.signature(loss_cls.__init__)
    forward_signature = inspect.signature(loss_cls.forward)

    init_param_names = {
        name for name, param in init_signature.parameters.items()
        if name not in {"self"} and param.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    forward_param_names = {
        name for name, param in forward_signature.parameters.items()
        if name not in {"self", "logits", "labels"} and param.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }

    init_kwargs = {}
    forward_kwargs = {}
    unused_kwargs = {}

    for key, value in kwargs.items():
        if key in {"logits", "labels"}:
            continue
        if key in forward_param_names:
            forward_kwargs[key] = value
        elif key in init_param_names:
            init_kwargs[key] = value
        else:
            unused_kwargs[key] = value

    return init_kwargs, forward_kwargs, unused_kwargs


class BaseLoss(nn.Module, ABC):
    def __init__(self, *args, reduction: str = "mean", ignore_index: int = IGNORE_INDEX, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of ['none', 'mean', 'sum']")

        self.reduction = reduction
        self.ignore_index = ignore_index

    def flatten(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, Any]:
        batch_size, seq_len = labels.shape

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

        return {
            "logits": shift_logits.view(-1, shift_logits.size(-1)),
            "labels": shift_labels.view(-1),
            "batch_size": batch_size,
            "seq_len": seq_len,
            **kwargs,
        }

    def build_valid_mask(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.ne(self.ignore_index)

    def reduce(
        self,
        loss: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.reduction == "none":
            return loss

        if self.reduction == "sum":
            reduced = loss.sum()
        else:
            if num_items_in_batch is not None:
                reduced = loss.sum() / num_items_in_batch
            elif valid_mask is None:
                reduced = loss.mean()
            else:
                valid_count = valid_mask.sum()
                if valid_count.item() == 0:
                    reduced = loss.sum() * 0.0
                else:
                    reduced = loss.sum() / valid_count

        return reduced

    @abstractmethod
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        raise NotImplementedError


class BaseMetricLoss(BaseLoss, ABC):
    scaled_metrics = {}

    @abstractmethod
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> LossOutput:
        raise NotImplementedError
