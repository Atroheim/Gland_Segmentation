import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    pred: raw logits (B,1,H,W)
    target: binary float (B,1,H,W)
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    denom = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = ((2.0 * inter + eps) / (denom + eps)).mean()
    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    pred: raw logits (B,1,H,W)
    target: binary float (B,1,H,W)
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = ((inter + eps) / (union + eps)).mean()
    return iou.item()
