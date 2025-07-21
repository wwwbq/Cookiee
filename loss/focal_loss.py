import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=2, alpha=10.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, input, target):
        logp = self.cross_entropy(input, target)
        p = torch.exp(-logp)
        loss = self.alpha * (1 - p) ** self.gamma * logp

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
