import copy
import time
import torch
from utils.losses import BCEDiceLoss
from utils.metrics import dice_score, iou_score


class FixMatchTrainer:
    """Mean Teacher variant of FixMatch for binary medical image segmentation.

    A EMA (exponential moving average) teacher model generates pseudo-labels
    for unlabeled images. Because the teacher is a slow-moving average of the
    student, its predictions are far more stable than the student's own
    predictions, breaking the unstable feedback loop that causes oscillation
    in vanilla FixMatch.

    Protocol:
      - Warmup: first `warmup_epochs` epochs are pure supervised on labeled data.
      - After warmup: student trains on both labeled (supervised) and unlabeled
        (teacher pseudo-label) data; teacher is updated via EMA each step.
      - No pixel-level confidence masking — image-level mean confidence gate.
      - BCEDiceLoss for both branches (preserves spatial structure).
      - Gradient clipping as a safety net.
    """

    def __init__(self, model, optimizer, device,
                 threshold: float = 0.7,
                 lambda_u: float = 0.5,
                 warmup_epochs: int = 20,
                 ema_decay: float = 0.999,
                 grad_clip: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.warmup_epochs = warmup_epochs
        self.ema_decay = ema_decay
        self.grad_clip = grad_clip
        self.criterion = BCEDiceLoss()
        self._epoch = 0
        self._global_step = 0

        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _update_teacher(self):
        decay = min(self.ema_decay, 1.0 - 1.0 / max(1, self._global_step))
        for tp, sp in zip(self.teacher.parameters(), self.model.parameters()):
            tp.data.mul_(decay).add_(sp.data * (1.0 - decay))

    def train_epoch(self, labeled_loader, unlabeled_loader) -> float:
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            lu = 0.0
        else:
            lu = self.lambda_u * min(1.0, (self._epoch - self.warmup_epochs) / 10.0)

        self.model.train()
        self.teacher.eval()
        total = 0.0
        n_batches = 0

        for (imgs_l, masks_l), (weak_u, strong_u) in zip(labeled_loader, unlabeled_loader):
            imgs_l, masks_l = imgs_l.to(self.device), masks_l.to(self.device)
            weak_u, strong_u = weak_u.to(self.device), strong_u.to(self.device)

            self.optimizer.zero_grad()

            loss_sup = self.criterion(self.model(imgs_l), masks_l)

            if lu > 0:
                with torch.no_grad():
                    pseudo_prob = torch.sigmoid(self.teacher(weak_u))

                pixel_conf = (pseudo_prob - 0.5).abs() * 2
                img_conf   = pixel_conf.mean(dim=(1, 2, 3))
                keep       = img_conf >= self.threshold

                if keep.any():
                    pseudo_lbl = (pseudo_prob[keep] > 0.5).float()
                    loss_u = self.criterion(self.model(strong_u[keep]), pseudo_lbl)
                    total_loss = loss_sup + lu * loss_u
                else:
                    total_loss = loss_sup
            else:
                total_loss = loss_sup

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self._update_teacher()

            total += total_loss.item()
            n_batches += 1
            self._global_step += 1

        return total / max(n_batches, 1)

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
            'iou':  sum(iou_vals)  / len(iou_vals),
            'time_ms': sum(times)  / len(times),
        }
