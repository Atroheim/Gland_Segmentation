import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)
        p = torch.sigmoid(pred)
        inter = (p * target).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * inter + self.eps) / (denom + self.eps)).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss
