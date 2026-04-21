import torch
from torch.nn import functional as F

from .base import BaseMetricLoss, LossOutput


class PreferenceLoss(BaseMetricLoss):
    scaled_metrics = {
        "ar_loss": True,
    }

    def __init__(
            self,
            balance_weight=0.005,
            beta=0.2,
            enable_AR=True,
            reduction="mean",
            **kwargs
    ) -> None:
        super().__init__(reduction=reduction, **kwargs)
        self.balance_weight = balance_weight
        self.beta = beta
        self.enable_AR = enable_AR
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction=reduction,
        )

    def get_batch_logps(self, logits: torch.Tensor, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        log_probs = logits.log_softmax(-1)
        per_token_logps = torch.gather(
            log_probs,
            dim=1,
            index=labels.unsqueeze(1).clamp(min=0),
        ).squeeze(1)

        mask = self.build_valid_mask(labels).float()
        masked_logps = per_token_logps * mask
        return masked_logps.view(batch_size, -1).sum(dim=-1)

    def dpo_loss(self, logits: torch.Tensor, labels: torch.Tensor, batch_size: int):
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even for Chosen/Rejected pairs.")

        policy_logps = self.get_batch_logps(logits, labels, batch_size)
        pairwise_batch_size = batch_size // 2

        policy_chosen_logps = policy_logps[:pairwise_batch_size]
        policy_rejected_logps = policy_logps[pairwise_batch_size:]
        margin = (policy_chosen_logps - policy_rejected_logps).mean()

        log_ratios = policy_chosen_logps - policy_rejected_logps
        log_ratios = torch.clamp(log_ratios, max=40.0)
        losses = -F.logsigmoid(self.beta * log_ratios)
        return losses.mean(), margin

    def ar_loss(self, logits: torch.Tensor, labels: torch.Tensor, num_items_in_batch=None):
        ce_loss = self.ce_loss(logits, labels)
        if num_items_in_batch is not None:
            assert self.ce_loss.reduction == "sum", "Reduction must be 'sum' for AR loss when num_items_in_batch is provided."
            ce_loss = ce_loss / num_items_in_batch
        return ce_loss

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, num_items_in_batch=None, **kwargs) -> LossOutput:
        flattened = self.flatten(logits, labels, **kwargs)
        logits = flattened["logits"]
        labels = flattened["labels"]
        batch_size = flattened["batch_size"]

        dpo_loss, margin = self.dpo_loss(logits, labels, batch_size)
        loss = dpo_loss
        scalar_metrics = {"margin": margin}

        if self.enable_AR:
            mid_point = logits.shape[0] // 2
            ar_loss = self.ar_loss(logits[:mid_point], labels[:mid_point], num_items_in_batch)

            scalar_metrics["ar_loss"] = ar_loss
            scalar_metrics["dpo_loss"] = dpo_loss
            loss = ar_loss + self.balance_weight * dpo_loss

        return LossOutput(
            loss=loss,
            scalar_metrics=scalar_metrics,
        )
