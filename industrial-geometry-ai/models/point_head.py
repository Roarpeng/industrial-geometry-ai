"""Head Fusion + Per-type Dense Point Heads（V4 第 12/14 节）。

Head Feature = concat(P3 上采样×2, stride-4 浅层 skip) → 3×3 → 128
Heads（每 type 一组，通道查表见 task/vocabulary.py）：
  heatmap 4ch（logits→sigmoid）  offset 8ch（tanh）
  size    4ch（sigmoid）          objectness 4ch（GAP→MLP→sigmoid）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadFusion(nn.Module):
    def __init__(self, p3_dim=128, skip_in=48, skip_dim=64):
        super().__init__()
        self.p3_proj = nn.Conv2d(p3_dim, p3_dim, 1)
        self.skip_proj = nn.Conv2d(skip_in, skip_dim, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(p3_dim + skip_dim, p3_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(p3_dim), nn.ReLU(inplace=True))

    def forward(self, p3, skip_s4):
        up = F.interpolate(self.p3_proj(p3), scale_factor=2, mode="bilinear",
                           align_corners=False)
        if up.shape[-2:] != skip_s4.shape[-2:]:  # 对齐舍入误差
            up = F.interpolate(up, size=skip_s4.shape[-2:], mode="bilinear",
                               align_corners=False)
        return self.fuse(torch.cat([up, self.skip_proj(skip_s4)], dim=1))


class PointHeads(nn.Module):
    def __init__(self, num_types=4, hidden=128, obj_hidden=64):
        super().__init__()
        self.num_types = num_types
        self.heat = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_types, 1))
        self.offset = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * num_types, 1))
        self.size = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_types, 1))
        self.objectness = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hidden, obj_hidden), nn.ReLU(inplace=True),
            nn.Linear(obj_hidden, num_types))
        # CenterNet 惯例：heatmap 偏置初始化为低值，避免初始全图 ~0.5 的负样本 loss 爆炸
        nn.init.constant_(self.heat[-1].bias, -4.0)

    def forward(self, feat):
        return {
            "heatmap": self.heat(feat),
            "offset": torch.tanh(self.offset(feat)),
            "size": torch.sigmoid(self.size(feat)),
            "objectness": self.objectness(feat),
        }
