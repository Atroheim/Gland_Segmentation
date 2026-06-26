import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from utils.transforms import WeakAugment


class GlaSDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, split='train', img_size=512,
                 n_samples=None, augment=False):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.augment = augment
        self.aug = WeakAugment() if augment else None

        files = sorted(f for f in os.listdir(img_dir) if f.startswith(split))
        if n_samples is not None:
            files = files[:n_samples]
        self.images = files

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        lbl_name = name.replace('.bmp', '_anno.bmp')

        img = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        lbl = Image.open(os.path.join(self.lbl_dir, lbl_name)).convert('L')

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        lbl = lbl.resize((self.img_size, self.img_size), Image.NEAREST)

        img_t = self.normalize(self.to_tensor(img))
        lbl_t = torch.from_numpy((np.array(lbl) > 0).astype(np.float32)).unsqueeze(0)

        if self.augment and self.aug is not None:
            img_t, lbl_t = self.aug(img_t, lbl_t)

        return img_t, lbl_t


class UnlabeledGlaSDataset(Dataset):
    def __init__(self, img_dir, split='train', img_size=512, start_idx=40):
        self.img_dir = img_dir
        self.img_size = img_size

        files = sorted(f for f in os.listdir(img_dir) if f.startswith(split))
        self.images = files[start_idx:]

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.images[idx])).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return self.normalize(self.to_tensor(img))
