#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import GlaSDataset, UnlabeledGlaSDataset
from models.unet import UNet
from trainers.supervised import SupervisedTrainer


CONFIG_MAP = {
    'unet':     'configs/unet.yaml',
    'unet40':   'configs/unet40.yaml',
    'sam':      'configs/sam.yaml',
    'medsam':   'configs/medsam.yaml',
    'fixmatch': 'configs/fixmatch.yaml',
}


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device):
    mtype = cfg['model']['type']
    if mtype == 'unet':
        return UNet(
            in_channels=cfg['model']['in_channels'],
            out_channels=cfg['model']['out_channels'],
            features=cfg['model']['features'],
        ).to(device)
    elif mtype == 'sam':
        from models.sam_seg import SAMSeg
        return SAMSeg(
            checkpoint=cfg['model']['checkpoint'],
            model_type=cfg['model']['model_type'],
            img_size=cfg['data']['img_size'],
        ).to(device)
    elif mtype == 'medsam':
        from models.medsam_seg import MedSAMSeg
        return MedSAMSeg(
            checkpoint=cfg['model']['checkpoint'],
            img_size=cfg['data']['img_size'],
        ).to(device)
    raise ValueError(f"Unknown model type: {mtype}")


def train_supervised(args, cfg, device):
    train_ds = GlaSDataset(
        cfg['data']['img_dir'], cfg['data']['lbl_dir'],
        split='train', img_size=cfg['data']['img_size'],
        n_samples=cfg['data'].get('n_samples'),
        augment=True,
    )
    val_ds = GlaSDataset(
        cfg['data']['img_dir'], cfg['data']['lbl_dir'],
        split='testA', img_size=cfg['data']['img_size'],
    )
    train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'],
                               shuffle=True, num_workers=cfg['data']['num_workers'],
                               pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                             num_workers=cfg['data']['num_workers'])

    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['train']['lr'],
    )
    trainer = SupervisedTrainer(model, optimizer, device, args.model)

    save_dir = cfg['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, cfg['train']['epochs'] + 1):
        loss = trainer.train_epoch(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            m = trainer.evaluate(val_loader)
            print(f"Epoch {epoch:3d}/{cfg['train']['epochs']} | "
                  f"loss={loss:.4f} dice={m['dice']:.4f} iou={m['iou']:.4f} "
                  f"time={m['time_ms']:.1f}ms")
            if m['dice'] > best_dice:
                best_dice = m['dice']
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))

    print(f"Best Dice: {best_dice:.4f}. Checkpoint: {save_dir}/best.pth")


def train_fixmatch(cfg, device):
    from trainers.fixmatch import FixMatchTrainer

    labeled_ds = GlaSDataset(
        cfg['data']['img_dir'], cfg['data']['lbl_dir'],
        split='train', img_size=cfg['data']['img_size'],
        n_samples=cfg['data']['n_labeled'], augment=True,
    )
    unlabeled_ds = UnlabeledGlaSDataset(
        cfg['data']['img_dir'], split='train',
        img_size=cfg['data']['img_size'],
        start_idx=cfg['data']['n_labeled'],
    )
    val_ds = GlaSDataset(
        cfg['data']['img_dir'], cfg['data']['lbl_dir'],
        split='testA', img_size=cfg['data']['img_size'],
    )
    labeled_loader = DataLoader(labeled_ds, batch_size=cfg['data']['labeled_batch_size'],
                                 shuffle=True, num_workers=cfg['data']['num_workers'],
                                 pin_memory=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=cfg['data']['unlabeled_batch_size'],
                                   shuffle=True, num_workers=cfg['data']['num_workers'],
                                   pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=cfg['data']['num_workers'])

    model = UNet(
        in_channels=cfg['model']['in_channels'],
        out_channels=cfg['model']['out_channels'],
        features=cfg['model']['features'],
    ).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['train']['lr'])
    trainer = FixMatchTrainer(
        model, optimizer, device,
        threshold=cfg['train']['threshold'],
        lambda_u=cfg['train']['lambda_u'],
    )

    save_dir = cfg['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, cfg['train']['epochs'] + 1):
        loss = trainer.train_epoch(labeled_loader, unlabeled_loader)
        if epoch % 10 == 0 or epoch == 1:
            m = trainer.evaluate(val_loader)
            print(f"Epoch {epoch:3d}/{cfg['train']['epochs']} | "
                  f"loss={loss:.4f} dice={m['dice']:.4f} iou={m['iou']:.4f}")
            if m['dice'] > best_dice:
                best_dice = m['dice']
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))

    print(f"Best Dice: {best_dice:.4f}. Checkpoint: {save_dir}/best.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(CONFIG_MAP.keys()), required=True)
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config or CONFIG_MAP[args.model])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {args.model} on {device}")

    if args.model == 'fixmatch':
        train_fixmatch(cfg, device)
    else:
        train_supervised(args, cfg, device)


if __name__ == '__main__':
    main()
