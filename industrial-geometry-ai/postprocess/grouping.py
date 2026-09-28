"""Object Grouping（V4 第 22 节）：

第一步：offset 投票 center 的贪心种子聚类（每组成员数封顶 K）
第二步：K 约束修正（缺员组从超员组按中心距离就近补齐，≤3 轮）
附加：双向一致性校验（|c_i − c_j| ≤ 0.25 × max(s_i, s_j)）
"""

from __future__ import annotations

import math
from typing import List, Tuple

from .peak_decode import PointDet


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def group_points(points: List[PointDet], k: int, gcfg: dict,
                 diag_px: float) -> Tuple[List[List[PointDet]], List[PointDet]]:
    """返回 (对象组列表, 游离点)。"""
    if not points:
        return [], []
    s_px = sorted(p.s * diag_px for p in points)
    s_med = s_px[len(s_px) // 2]
    seed_tol = gcfg.get("seed_tol_factor", 0.35) * s_med
    bidir_tol = gcfg.get("bidir_tol_factor", 0.25)
    rounds = gcfg.get("correction_rounds", 3)

    order = sorted(range(len(points)), key=lambda i: -points[i].conf)
    assigned = [False] * len(points)
    groups: List[List[int]] = []

    # ---- 第一步：种子聚类（封顶 K）----
    for i in order:
        if assigned[i]:
            continue
        seed = points[i]
        grp = [i]
        assigned[i] = True
        for j in order:
            if len(grp) >= k:
                break
            if assigned[j]:
                continue
            q = points[j]
            if _dist(q.cx, q.cy, seed.cx, seed.cy) > seed_tol:
                continue
            # 双向一致性：与组内至少一个成员的投票中心距离 ≤ tol × max(size)
            ok = any(_dist(q.cx, q.cy, points[m].cx, points[m].cy)
                     <= bidir_tol * max(q.s, points[m].s) * diag_px for m in grp)
            if not ok:
                continue
            assigned[j] = True
            grp.append(j)
        groups.append(grp)

    # ---- 第二步：K 约束修正 ----
    def group_center(g: List[int]):
        return (sum(points[i].cx for i in g) / len(g),
                sum(points[i].cy for i in g) / len(g))

    for _ in range(rounds):
        moved = False
        for gi, g in enumerate(groups):
            if len(g) >= k:
                continue
            gx, gy = group_center(g)
            for gj, src in enumerate(groups):
                if gi == gj or len(src) <= k:
                    continue
                # 超员组让出离缺员组中心最近的成员
                cands = sorted(src[k:], key=lambda i: _dist(points[i].cx, points[i].cy, gx, gy))
                for i in cands:
                    if len(g) >= k:
                        break
                    if _dist(points[i].cx, points[i].cy, gx, gy) <= 1.0 * s_med:
                        src.remove(i)
                        g.append(i)
                        moved = True
            if moved:
                break  # 重新计算各中心后再来一轮
        if not moved:
            break

    # ---- 游离点：就近并入还有空位的组 ----
    for i in range(len(points)):
        if assigned[i]:
            continue
        p = points[i]
        best, best_d = None, 1.0 * s_med
        for g in groups:
            if len(g) >= k:
                continue
            gx, gy = group_center(g)
            d = _dist(p.cx, p.cy, gx, gy)
            if d < best_d:
                best, best_d = g, d
        if best is not None:
            best.append(i)
            assigned[i] = True

    leftovers = [points[i] for i in range(len(points)) if not assigned[i]]
    return [[points[i] for i in g] for g in groups], leftovers
