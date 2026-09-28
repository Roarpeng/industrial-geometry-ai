"""CenterNet Focal Loss + per-type dense 归一化（V4 第 25 节）。"""

from __future__ import annotations

import torch


def heatmap_focal_loss(pred_logits: torch.Tensor, gt: torch.Tensor,
                       alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """pred_logits/gt: (B, T, Hc, Wc)。每 type 通道独立按正样本数归一化。"""
    p = pred_logits.sigmoid()
    pos = gt == 1
    ce_pos = -((1 - p) ** alpha) * torch.log(p.clamp_min(1e-9))
    ce_neg = -((1 - gt) ** beta) * (p ** alpha) * torch.log((1 - p).clamp_min(1e-9))
    loss_map = torch.where(pos, ce_pos, ce_neg)

    n_pos = pos.sum(dim=(0, 2, 3)).clamp_min(1.0)      # dense 归一化（V4 第 25 节）
    per_type = loss_map.sum(dim=(0, 2, 3)) / n_pos
    return per_type.mean()
