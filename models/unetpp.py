import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from models.unet import DoubleConv


class UNetPP(nn.Module):
    """UNet++ with full dense skip connections (no deep supervision)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 features: List[int] = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        n = len(features)
        self.n = n
        self.features = features
        bottleneck_ch = features[-1] * 2
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoder.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], bottleneck_ch)

        self.node_convs = nn.ModuleList()
        for i in range(n):
            row = nn.ModuleList()
            below_ch = bottleneck_ch if i == n - 1 else features[i + 1]
            for j in range(1, n - i + 1):
                in_ch = features[i] * j + below_ch
                row.append(DoubleConv(in_ch, features[i]))
            self.node_convs.append(row)

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n
        nodes = {}

        feat = x
        for i in range(n):
            nodes[(i, 0)] = self.encoder[i](feat)
            feat = self.pool(nodes[(i, 0)])
        nodes[(n, 0)] = self.bottleneck(feat)

        for j in range(1, n + 1):
            for i in range(n - j + 1):
                prev_same = [nodes[(i, k)] for k in range(j)]
                below = nodes[(i + 1, j - 1)]
                below_up = F.interpolate(
                    below, size=prev_same[0].shape[2:],
                    mode='bilinear', align_corners=False,
                )
                inp = torch.cat(prev_same + [below_up], dim=1)
                nodes[(i, j)] = self.node_convs[i][j - 1](inp)

        return self.final(nodes[(0, n)])
