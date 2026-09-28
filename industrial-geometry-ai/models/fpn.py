"""Lite FPN（V4 第 12 节）：P3/P4/P5，输出 P3（stride 8）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteFPN(nn.Module):
    def __init__(self, in_dims=(96, 192, 384), out_dim: int = 128):
        super().__init__()
        self.lat8 = nn.Conv2d(in_dims[0], out_dim, 1)
        self.lat16 = nn.Conv2d(in_dims[1], out_dim, 1)
        self.lat32 = nn.Conv2d(in_dims[2], out_dim, 1)
        self.smooth = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True))

    def forward(self, feats):
        p5 = self.lat32(feats["s32"])
        p4 = self.lat16(feats["s16"]) + F.interpolate(
            p5, size=feats["s16"].shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lat8(feats["s8"]) + F.interpolate(
            p4, size=feats["s8"].shape[-2:], mode="bilinear", align_corners=False)
        return self.smooth(p3)
