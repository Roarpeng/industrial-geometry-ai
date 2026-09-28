"""四类图形的纯几何工具（Geometry Engine 子模块，V4 第 57 节目录）。"""

from __future__ import annotations

import math
from typing import List, Tuple

Pt = Tuple[float, float]


def centroid(pts: List[Pt]) -> Pt:
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def polygon_area(pts: List[Pt]) -> float:
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def segment_length(p1: Pt, p2: Pt) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def angle_of(p1: Pt, p2: Pt) -> float:
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])
