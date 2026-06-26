import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import pytest
from models.unet import UNet


def test_unet_output_shape_512():
    model = UNet()
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.shape == (2, 1, 512, 512), f"Got {out.shape}"

def test_unet_output_shape_256():
    model = UNet()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 1, 256, 256)

def test_unet_no_nan():
    model = UNet()
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    assert not torch.isnan(out).any(), "NaN in UNet output"

def test_unet_grad_flows():
    model = UNet()
    x = torch.randn(1, 3, 64, 64, requires_grad=False)
    out = model(x)
    loss = out.mean()
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
