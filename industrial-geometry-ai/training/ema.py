"""EMA 权重（V4 第 43 节：验证/导出一律用 EMA 权重）。"""

from __future__ import annotations

import copy

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.base_decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        # YOLO 式 warmup：早期 decay 小，EMA 快速跟上当前权重，
        # 否则 0.9999 衰减下前几千步 EMA 几乎仍是初始化权重
        self.updates += 1
        d = min(self.base_decay, (1.0 + self.updates) / (10.0 + self.updates))
        msd = model.state_dict()
        for k, e in self.module.state_dict().items():
            m = msd[k]
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                e.copy_(m)
