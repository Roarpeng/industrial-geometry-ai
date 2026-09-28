"""缺角几何补全（V4 第 24 节）。

只处理"恰好缺一个点"的组（len == K−1）：
    Line:      p2 = 2c − p1                          （中点关系）
    Triangle:  p3 = 3c − p1 − p2                     （质心关系）
    Rectangle: p4 = A + C − B                        （平行四边形，A/C 为对角点）
    Star:      极坐标拟合（36° 等角 + 内外交替半径）
center 来自组内 offset 投票的加权平均。补全点：
    predicted=True，confidence = 组内最低 × 0.9，ROI × 1.5。
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .peak_decode import PointDet


def _voted_center(grp: List[PointDet]):
    w = np.asarray([p.conf for p in grp], dtype=np.float64)
    w = np.clip(w, 1e-6, None) / np.clip(w, 1e-6, None).sum()
    cx = sum(wi * p.cx for wi, p in zip(w, grp))
    cy = sum(wi * p.cy for wi, p in zip(w, grp))
    return cx, cy


def _complete_star(pts: np.ndarray, c) -> Optional[np.ndarray]:
    rel = pts - np.asarray(c)
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    r = np.hypot(rel[:, 0], rel[:, 1])
    order = np.argsort(ang)
    ang_s, r_s = ang[order], r[order]
    gaps = np.diff(np.concatenate([ang_s, ang_s[:1] + 2 * math.pi]))
    expected = 2 * math.pi / 10
    gi = int(np.argmax(gaps))
    if gaps[gi] < 1.5 * expected:
        return None  # 没有明显缺角
    a_miss = ang_s[gi] + gaps[gi] / 2.0
    # 半径类别：内外顶点按中位数二分；缺口两邻同类 → 缺点为另一类
    med = np.median(r_s)
    prev_r, next_r = r_s[gi], r_s[(gi + 1) % len(r_s)]
    if prev_r > med and next_r > med:
        r_miss = r_s[r_s <= med].mean() if (r_s <= med).any() else med
    elif prev_r < med and next_r < med:
        r_miss = r_s[r_s > med].mean() if (r_s > med).any() else med
    else:
        r_miss = float(med)
    return np.asarray(c) + r_miss * np.asarray([math.cos(a_miss), math.sin(a_miss)])


def complete_group(grp: List[PointDet], k: int, type_name: str) -> Optional[PointDet]:
    """恰好缺一个点时返回补全点，否则 None。"""
    if len(grp) != k - 1:
        return None
    cx, cy = _voted_center(grp)
    pts = np.asarray([[p.x, p.y] for p in grp], dtype=np.float64)

    if type_name == "line":
        new = 2 * np.asarray([cx, cy]) - pts[0]
    elif type_name == "triangle":
        new = 3 * np.asarray([cx, cy]) - pts[0] - pts[1]
    elif type_name == "rectangle":
        d = [math.dist(pts[0], pts[1]), math.dist(pts[0], pts[2]), math.dist(pts[1], pts[2])]
        diag_pair = max(range(3), key=lambda i: d[i])  # 最大的边对应两个对角点
        idx = [(0, 1), (0, 2), (1, 2)][diag_pair]
        a, c_ = pts[idx[0]], pts[idx[1]]
        b = pts[3 - idx[0] - idx[1]]
        new = a + c_ - b
    elif type_name == "star":
        new = _complete_star(pts, (cx, cy))
        if new is None:
            return None
    else:
        return None

    base = min(grp, key=lambda p: p.conf)
    return PointDet(
        type_idx=base.type_idx, x=float(new[0]), y=float(new[1]),
        conf=base.conf * 0.9, s=base.s, roi=base.roi * 1.5,
        cx=cx, cy=cy, predicted=True)
