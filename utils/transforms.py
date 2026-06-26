import random
import torch
import torchvision.transforms.functional as TF


class WeakAugment:
    """Synchronized random flips and rotation for image+mask."""

    def __call__(self, img, mask):
        if random.random() > 0.5:
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            if mask is not None:
                mask = TF.vflip(mask)
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle)
        if mask is not None:
            mask = TF.rotate(mask, angle)
        return img, mask


def cutout(img, length=128):
    """Zero out a random square patch in img (in-place on a clone)."""
    img = img.clone()
    h, w = img.shape[-2], img.shape[-1]
    y = random.randint(0, h - 1)
    x = random.randint(0, w - 1)
    y1, y2 = max(0, y - length // 2), min(h, y + length // 2)
    x1, x2 = max(0, x - length // 2), min(w, x + length // 2)
    img[:, y1:y2, x1:x2] = 0.0
    return img


class StrongAugment:
    """Strong augmentation for unlabeled images (no mask needed)."""

    def __call__(self, img, mask=None):
        if random.random() > 0.5:
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            if mask is not None:
                mask = TF.vflip(mask)

        # Color jitter on image only
        img = TF.adjust_brightness(img, 0.6 + random.random() * 0.8)
        img = TF.adjust_contrast(img, 0.6 + random.random() * 0.8)
        img = TF.adjust_saturation(img, 0.6 + random.random() * 0.8)

        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle)
        if mask is not None:
            mask = TF.rotate(mask, angle)

        img = cutout(img, length=img.shape[-1] // 4)
        return img, mask
