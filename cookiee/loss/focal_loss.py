import torch

from .base import BaseLoss


class FocalLoss(BaseLoss):
    def __init__(self, weight=None, reduction="mean", gamma=2, alpha=10.0, eps=1e-7, **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, num_items_in_batch=None, **kwargs):
        flattened = self.flatten(logits, labels, **kwargs)
        logits = flattened["logits"]
        labels = flattened["labels"]

        ce_loss = self.cross_entropy(logits, labels)
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        valid_mask = self.build_valid_mask(labels)
        return self.reduce(
            focal_loss,
            valid_mask=valid_mask,
            num_items_in_batch=num_items_in_batch,
        )
