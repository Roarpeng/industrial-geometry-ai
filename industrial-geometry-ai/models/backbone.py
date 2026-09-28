"""轻量 Backbone（V4 第 11 节）：输出 s4/s8/s16/s32 四级特征。"""

from __future__ import annotations

import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class Backbone(nn.Module):
    """4 个 stage，每个 stage 两层 3×3，首层下采样。"""

    def __init__(self, channels=(48, 96, 192, 384)):
        super().__init__()
        c0 = channels[0] // 2
        self.stem = ConvBNReLU(3, c0, k=3, s=2)          # s2
        self.s4 = nn.Sequential(ConvBNReLU(c0, channels[0], s=2),
                                ConvBNReLU(channels[0], channels[0]))
        self.s8 = nn.Sequential(ConvBNReLU(channels[0], channels[1], s=2),
                                ConvBNReLU(channels[1], channels[1]))
        self.s16 = nn.Sequential(ConvBNReLU(channels[1], channels[2], s=2),
                                 ConvBNReLU(channels[2], channels[2]))
        self.s32 = nn.Sequential(ConvBNReLU(channels[2], channels[3], s=2),
                                 ConvBNReLU(channels[3], channels[3]))

    def forward(self, x):
        x = self.stem(x)
        f4 = self.s4(x)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        return {"s4": f4, "s8": f8, "s16": f16, "s32": f32}
