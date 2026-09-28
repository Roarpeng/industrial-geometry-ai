"""Offset / Size / Objectness 损失（V4 第 27/28 节）。

offset 与 size 只在 heatmap GT 峰值 cell 处监督（mask）。
size 在 log 空间回归（跨度大）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def offset_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/gt: (B, 2T, Hc, Wc)；mask: (B, T, Hc, Wc)。"""
    m = mask.repeat_interleave(2, dim=1)
    loss = F.smooth_l1_loss(pred, gt, reduction="none")
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def size_log_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/gt/mask: (B, T, Hc, Wc)。"""
    loss = F.smooth_l1_loss((pred + EPS).log(), (gt + EPS).log(), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def objectness_loss(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred_logits, gt)
