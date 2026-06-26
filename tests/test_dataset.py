import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import pytest
from data.dataset import GlaSDataset, UnlabeledGlaSDataset

IMG_DIR = '/root/autodl-tmp/Glas/Image'
LBL_DIR = '/root/autodl-tmp/Glas/Label'

def test_train_length():
    ds = GlaSDataset(IMG_DIR, LBL_DIR, split='train')
    assert len(ds) == 85

def test_testa_length():
    ds = GlaSDataset(IMG_DIR, LBL_DIR, split='testA')
    assert len(ds) == 60

def test_testb_length():
    ds = GlaSDataset(IMG_DIR, LBL_DIR, split='testB')
    assert len(ds) == 20

def test_item_shapes_512():
    ds = GlaSDataset(IMG_DIR, LBL_DIR, split='train', img_size=512)
    img, lbl = ds[0]
    assert img.shape == (3, 512, 512), f"Expected (3,512,512) got {img.shape}"
    assert lbl.shape == (1, 512, 512), f"Expected (1,512,512) got {lbl.shape}"

def test_label_binary():
    ds = GlaSDataset(IMG_DIR, LBL_DIR, split='train')
    _, lbl = ds[0]
    unique = set(lbl.unique().tolist())
    assert unique.issubset({0.0, 1.0}), f"Non-binary label values: {unique}"

def test_img_dtype():
    ds = GlaSDataset(IMG_DIR, LBL_DIR, split='train')
    img, _ = ds[0]
    assert img.dtype == torch.float32

def test_n_samples():
    ds = GlaSDataset(IMG_DIR, LBL_DIR, split='train', n_samples=40)
    assert len(ds) == 40

def test_unlabeled_dataset():
    ds = UnlabeledGlaSDataset(IMG_DIR, split='train', img_size=512, start_idx=40)
    assert len(ds) == 45
    img = ds[0]
    assert img.shape == (3, 512, 512)
