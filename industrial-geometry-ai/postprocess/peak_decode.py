"""Peak Decode（V4 第 21 节）：

3×3 max-pool NMS → 阈值 τ → top-K → 亚像素细化（3×3 加权质心，w = heat^4）
→ 读 offset/size → 解析式 ROI → offset 投票 center。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F

STRIDE = 4


@dataclass
class PointDet:
    type_idx: int
    x: float              # 亚像素细化后（输入图像像素坐标）
    y: float
    conf: float
    s: float              # 归一化 bbox 对角线（/图像对角线）
    roi: float            # 归一化 ROI 半径
    cx: float             # offset 投票的对象中心（像素）
    cy: float
    predicted: bool = False  # 几何补全点（V4 第 24 节）


def _subpixel(h: torch.Tensor, y: int, x: int, power: float):
    """3×3 邻域加权质心（w = heat^power）。"""
    hp = F.pad(h[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    win = hp[y:y + 3, x:x + 3]
    w = win.clamp_min(0).pow(power)
    tot = w.sum().clamp_min(1e-12)
    dx = float((w * torch.tensor([[0., 1., 2.]] * 3, device=h.device)).sum() / tot - 1.0)
    dy = float((w * torch.tensor([[0.], [1.], [2.]], device=h.device)).sum() / tot - 1.0)
    return dx, dy


def decode_batch(outputs: dict, dcfg: dict, W: int, H: int) -> List[List[PointDet]]:
    """outputs: 模型输出 dict（batch 维在前）。返回每张图的 PointDet 列表（全部 type）。"""
    heat = outputs["heatmap"].sigmoid().detach().cpu()
    off = outputs["offset"].detach().cpu()
    size = outputs["size"].detach().cpu()

    tau = dcfg["peak_threshold"]
    topk = dcfg.get("topk", 1000)
    power = dcfg.get("subpixel_power", 4)
    roi_ratio = dcfg.get("roi", {}).get("ratio", 0.10)
    min_r = dcfg.get("roi", {}).get("min_r", 0.01)
    max_r = dcfg.get("roi", {}).get("max_r", 0.20)
    diag_px = math.hypot(W, H)

    B, T, Hc, Wc = heat.shape
    results: List[List[PointDet]] = []
    for b in range(B):
        dets: List[PointDet] = []
        for t in range(T):
            h = heat[b, t]
            mp = F.max_pool2d(h[None, None], 3, stride=1, padding=1)[0, 0]
            peak = (h >= mp) & (h >= tau)
            ys, xs = torch.nonzero(peak, as_tuple=True)
            if ys.numel() == 0:
                continue
            scores = h[ys, xs]
            order = scores.argsort(descending=True)[:topk]
            for i in order.tolist():
                yy, xx = int(ys[i]), int(xs[i])
                dx, dy = _subpixel(h, yy, xx, power)
                px = (xx + dx + 0.5) * STRIDE
                py = (yy + dy + 0.5) * STRIDE
                ox = float(off[b, 2 * t, yy, xx])
                oy = float(off[b, 2 * t + 1, yy, xx])
                s = float(size[b, t, yy, xx])
                s_px = s * diag_px
                roi = float(min(max(roi_ratio * s, min_r), max_r))
                dets.append(PointDet(
                    type_idx=t, x=px, y=py, conf=float(scores[i]), s=s, roi=roi,
                    cx=px + ox * s_px, cy=py + oy * s_px))
        results.append(dets)
    return results
