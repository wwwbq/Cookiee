import torch
from torch import nn

from .base import BaseMetricLoss, LossOutput


class DomainLoss(BaseMetricLoss):
    def __init__(self, domain_mapping, reduction="mean", **kwargs) -> None:
        super().__init__(reduction=reduction, **kwargs)
        self.domain_mapping = domain_mapping
        self.loss_func = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction="none",
        )
        #self.loss_func = nn.functional.cross_entropy()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, domains, num_items_in_batch=None, **kwargs) -> LossOutput:
        flattened = self.flatten(logits, labels, **kwargs)
        logits = flattened["logits"]
        labels = flattened["labels"]
        seq_len = flattened["seq_len"]

        domains = domains.to(labels.device).repeat_interleave(seq_len - 1)
        assert logits.shape[0] == labels.shape[0] == domains.shape[0]

        loss_per_token = self.loss_func(logits, labels)
        valid_mask = self.build_valid_mask(labels)
        loss = self.reduce(
            loss_per_token,
            valid_mask=valid_mask,
            num_items_in_batch=num_items_in_batch,
        )

        grouped_metrics = {"domain_loss": {}}
        for domain_id in torch.unique(domains.to(torch.int32)):
            domain_id_int = domain_id.item()
            domain_name = self.domain_mapping.get(domain_id_int, f"unknown_domain_{domain_id_int}")

            domain_mask = (domains == domain_id)
            mask = domain_mask & valid_mask

            grouped_metrics["domain_loss"][f"{domain_name}_loss"] = {
                "sum": loss_per_token[mask].sum(),
                "count": mask.sum().to(dtype=torch.int),
            }

        return LossOutput(
            loss=loss,
            grouped_metrics=grouped_metrics,
        )
