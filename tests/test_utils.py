import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import pytest
from utils.metrics import dice_score, iou_score
from utils.losses import BCEDiceLoss



def test_dice_perfect_overlap():
    pred = torch.full((1, 1, 4, 4), 10.0)
    target = torch.ones(1, 1, 4, 4)
    assert abs(dice_score(pred, target) - 1.0) < 1e-3

def test_dice_no_overlap():
    pred = torch.full((1, 1, 4, 4), 10.0)
    target = torch.zeros(1, 1, 4, 4)
    assert dice_score(pred, target) < 0.01

def test_iou_perfect_overlap():
    pred = torch.full((1, 1, 4, 4), 10.0)
    target = torch.ones(1, 1, 4, 4)
    assert abs(iou_score(pred, target) - 1.0) < 1e-3

def test_iou_no_overlap():
    pred = torch.full((1, 1, 4, 4), 10.0)
    target = torch.zeros(1, 1, 4, 4)
    assert iou_score(pred, target) < 0.01

def test_dice_returns_float():
    pred = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    result = dice_score(pred, target)
    assert isinstance(result, float)


def test_loss_is_scalar():
    loss_fn = BCEDiceLoss()
    pred = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()
    loss = loss_fn(pred, target)
    assert loss.shape == torch.Size([])

def test_loss_positive():
    loss_fn = BCEDiceLoss()
    pred = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()
    assert loss_fn(pred, target).item() > 0

def test_loss_perfect_prediction_low():
    loss_fn = BCEDiceLoss()
    pred = torch.full((1, 1, 4, 4), 10.0)
    target = torch.ones(1, 1, 4, 4)
    assert loss_fn(pred, target).item() < 0.1
