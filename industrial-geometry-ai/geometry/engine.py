"""Geometry Engine（V4 第 55 节）：模型学习视觉，程序执行数学。

不参与训练。负责：排序 / 等分 / 中点 / 中心 / 距离 / 角度 /
post_operation（divide 等）/ 关系筛选（largest / nearest …）。
"""

from __future__ import annotations

import math
from typing import Dict, List

from task.task_spec import TaskSpec


def order_clockwise(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """按绕质心的极角排序（输出 API 用，保证点序确定）。"""
    cx = sum(p["u"] for p in points) / len(points)
    cy = sum(p["v"] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p["v"] - cy, p["u"] - cx))


def divide_line(points: List[Dict[str, float]], n: int) -> List[Dict[str, float]]:
    """线段 n 等分 → n−1 个中间点（V4 第 6.2 节 post_operation）。"""
    assert len(points) == 2, "divide 只作用于 line 的两个端点"
    (x1, y1), (x2, y2) = (points[0]["u"], points[0]["v"]), (points[1]["u"], points[1]["v"])
    out = []
    for i in range(1, n):
        t = i / n
        out.append({"u": x1 + (x2 - x1) * t, "v": y1 + (y2 - y1) * t, "generated": "divide"})
    return out


def select_by_relation(objects: List[dict], key: str) -> List[dict]:
    """关系筛选（V4 第 38 节）：largest / smallest / top_left … 全部 CPU 计算。"""
    if not objects:
        return []
    def bbox_diag(o):
        pts = [(p["u"], p["v"]) for p in o["points"]]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    def top_left_score(o):
        return sum(p["u"] + p["v"] for p in o["points"]) / len(o["points"])
    if key == "largest":
        return [max(objects, key=bbox_diag)]
    if key == "smallest":
        return [min(objects, key=bbox_diag)]
    if key == "top_left":
        return [min(objects, key=top_left_score)]
    raise ValueError(f"unsupported relation: {key}")


def apply_post_operation(spec: TaskSpec, objects: List[dict]) -> List[dict]:
    post = spec.post_operation
    if not post:
        return objects
    if post.get("op") == "divide":
        for o in objects:
            mids = divide_line(o["points"], post.get("n", 2))
            o["derived_points"] = mids
        return objects
    raise ValueError(f"unsupported post_operation: {post}")
