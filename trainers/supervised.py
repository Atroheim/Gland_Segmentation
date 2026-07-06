import time
import torch
from utils.losses import BCEDiceLoss
from utils.metrics import dice_score, iou_score


class SupervisedTrainer:
    def __init__(self, model, optimizer, device, model_name: str):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.criterion = BCEDiceLoss()
        self._is_sam = model_name in ('sam', 'medsam')

    def train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(imgs)
            loss = self.criterion(preds, masks)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    def evaluate(self, loader) -> dict:
        self.model.eval()
        dice_vals, iou_vals, times = [], [], []
        with torch.no_grad():
            for imgs, masks in loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                t0 = time.time()
                preds = self.model(imgs)
                ms = (time.time() - t0) * 1000.0 / imgs.shape[0]
                times.append(ms)
                dice_vals.append(dice_score(preds, masks))
                iou_vals.append(iou_score(preds, masks))
        return {
            'dice': sum(dice_vals) / len(dice_vals),
            'iou': sum(iou_vals) / len(iou_vals),
            'time_ms': sum(times) / len(times),
        }
