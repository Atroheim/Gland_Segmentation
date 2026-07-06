#!/usr/bin/env python3
"""Evaluate all trained checkpoints on testA and testB splits."""
import argparse
import os
import time
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset import GlaSDataset
from utils.metrics import dice_score, iou_score


MODELS = {
    'unet':     ('configs/unet.yaml',     'results/unet/best.pth'),
    'unet40':   ('configs/unet40.yaml',   'results/unet40/best.pth'),
    'unetpp':   ('configs/unetpp.yaml',   'results/unetpp/best.pth'),
    'sam':      ('configs/sam.yaml',      'results/sam/best.pth'),
    'medsam':   ('configs/medsam.yaml',   'results/medsam/best.pth'),
    'fixmatch': ('configs/fixmatch.yaml', 'results/fixmatch/best.pth'),
}


def load_model(cfg: dict, ckpt: str, device):
    mtype = cfg['model']['type']
    if mtype == 'unet':
        from models.unet import UNet
        model = UNet(
            in_channels=cfg['model']['in_channels'],
            out_channels=cfg['model']['out_channels'],
            features=cfg['model']['features'],
        )
    elif mtype == 'unetpp':
        from models.unetpp import UNetPP
        model = UNetPP(
            in_channels=cfg['model']['in_channels'],
            out_channels=cfg['model']['out_channels'],
            features=cfg['model']['features'],
        )
    elif mtype == 'sam':
        from models.sam_seg import SAMSeg
        model = SAMSeg(
            checkpoint=cfg['model']['checkpoint'],
            model_type=cfg['model']['model_type'],
            img_size=cfg['data']['img_size'],
        )
    elif mtype == 'medsam':
        from models.medsam_seg import MedSAMSeg
        model = MedSAMSeg(
            checkpoint=cfg['model']['checkpoint'],
            img_size=cfg['data']['img_size'],
        )
    else:
        raise ValueError(f"Unknown model: {mtype}")

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def evaluate_split(model, split: str, cfg: dict, device, vis_dir: str = None, n_vis: int = 4):
    ds = GlaSDataset(
        cfg['data']['img_dir'], cfg['data']['lbl_dir'],
        split=split, img_size=cfg['data']['img_size'],
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    dice_vals, iou_vals, time_vals = [], [], []
    saved = 0

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(device), masks.to(device)
            t0 = time.perf_counter()
            preds = model(imgs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            dice_vals.append(dice_score(preds, masks))
            iou_vals.append(iou_score(preds, masks))
            time_vals.append(elapsed_ms)

            if vis_dir and saved < n_vis:
                _save_vis(imgs[0], masks[0], preds[0], os.path.join(vis_dir, f'{split}_{i}.png'))
                saved += 1

    return {
        'dice': float(np.mean(dice_vals)),
        'iou':  float(np.mean(iou_vals)),
        'time_ms': float(np.mean(time_vals)),
    }


def _save_vis(img, mask, pred, path):
    img_np = img.cpu().permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    mask_np  = mask.cpu().squeeze().numpy()
    pred_bin = (torch.sigmoid(pred) > 0.5).cpu().squeeze().float().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np);   axes[0].set_title('Image');     axes[0].axis('off')
    axes[1].imshow(mask_np,  cmap='gray'); axes[1].set_title('GT');   axes[1].axis('off')
    axes[2].imshow(pred_bin, cmap='gray'); axes[2].set_title('Pred'); axes[2].axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument('--splits', nargs='+', default=['testA', 'testB'])
    parser.add_argument('--vis', action='store_true', help='Save prediction visualizations')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    header = f"{'Model':<12}" + "".join(
        f"  {s} Dice  {s} IoU  {s} ms" for s in args.splits
    )
    print(header)
    print('-' * len(header))

    all_results = {}
    for name in args.models:
        cfg_path, ckpt = MODELS[name]
        if not os.path.exists(ckpt):
            print(f"{name:<12}  checkpoint not found: {ckpt}")
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        model = load_model(cfg, ckpt, device)

        row = f"{name:<12}"
        split_results = {}
        for split in args.splits:
            vis_dir = f'results/vis/{name}' if args.vis else None
            m = evaluate_split(model, split, cfg, device, vis_dir=vis_dir)
            split_results[split] = m
            row += f"  {m['dice']:.4f}  {m['iou']:.4f}  {m['time_ms']:6.1f}"
        print(row)
        all_results[name] = split_results

    os.makedirs('results', exist_ok=True)
    import csv
    with open('results/eval_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        cols = ['model'] + [f'{s}_{k}' for s in args.splits for k in ('dice', 'iou', 'time_ms')]
        writer.writerow(cols)
        for name, sres in all_results.items():
            row = [name]
            for s in args.splits:
                if s in sres:
                    row += [f"{sres[s]['dice']:.4f}", f"{sres[s]['iou']:.4f}", f"{sres[s]['time_ms']:.2f}"]
                else:
                    row += ['', '', '']
            writer.writerow(row)
    print("\nResults saved to results/eval_results.csv")

    if args.vis:
        print("Visualizations saved to results/vis/")


if __name__ == '__main__':
    main()
