"""几何图形参数随机化（V4 方案第 31 节）。

每个函数返回该对象的 K 个关键点（像素坐标）。
有效性硬约束（间距/重叠/边界）由 generate.py 的场景层统一裁决。
"""

from __future__ import annotations

import numpy as np

PI2 = 2.0 * np.pi


def _jitter(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def gen_triangle(rng: np.random.Generator) -> np.ndarray:
    """position / scale / rotation / aspect / vertex perturbation"""
    r = rng.uniform(22.0, 150.0)
    rot = rng.uniform(0.0, PI2)
    sx, sy = rng.uniform(0.7, 1.4), rng.uniform(0.7, 1.4)
    pts = []
    for k in range(3):
        ang = rot + k * (PI2 / 3.0) + np.deg2rad(rng.uniform(-15.0, 15.0))
        rad = r * rng.uniform(0.8, 1.2)
        pts.append((np.cos(ang) * rad * sx, np.sin(ang) * rad * sy))
    return np.asarray(pts, dtype=np.float64)


def gen_rectangle(rng: np.random.Generator) -> np.ndarray:
    """width / height / rotation / corner perturbation（perspective 留 Stage B）"""
    w = rng.uniform(40.0, 260.0)
    h = np.clip(w * rng.uniform(0.4, 2.2), 32.0, 300.0)
    rot = rng.uniform(0.0, np.pi)
    cos, sin = np.cos(rot), np.sin(rot)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    pts = []
    for x, y in corners:
        x += rng.uniform(-3.0, 3.0)
        y += rng.uniform(-3.0, 3.0)
        pts.append((x * cos - y * sin, x * sin + y * cos))
    return np.asarray(pts, dtype=np.float64)


def gen_star(rng: np.random.Generator) -> np.ndarray:
    """outer / inner radius / rotation / aspect / vertex jitter（inner 有下限保证星形）"""
    r_out = rng.uniform(26.0, 140.0)
    r_in = r_out * rng.uniform(0.42, 0.62)
    rot = rng.uniform(0.0, PI2)
    sx, sy = rng.uniform(0.75, 1.3), rng.uniform(0.75, 1.3)
    pts = []
    for k in range(10):
        ang = rot + k * (np.pi / 5.0)
        rad = r_out if k % 2 == 0 else r_in
        rad *= rng.uniform(0.97, 1.03)
        pts.append((np.cos(ang) * rad * sx, np.sin(ang) * rad * sy))
    return np.asarray(pts, dtype=np.float64)


def gen_line(rng: np.random.Generator) -> np.ndarray:
    """length / angle / curvature（端点即关键点，curvature 只影响渲染）"""
    length = rng.uniform(60.0, 380.0)
    ang = rng.uniform(0.0, np.pi)
    dx, dy = np.cos(ang) * length / 2, np.sin(ang) * length / 2
    p1 = np.array([-dx + _jitter(rng, -2, 2), -dy + _jitter(rng, -2, 2)])
    p2 = np.array([dx + _jitter(rng, -2, 2), dy + _jitter(rng, -2, 2)])
    return np.stack([p1, p2])


GENERATORS = {
    "triangle": gen_triangle,
    "rectangle": gen_rectangle,
    "star": gen_star,
    "line": gen_line,
}


def gen_object(rng: np.random.Generator, obj_type: str, w: int, h: int, margin: float):
    """生成一个对象并平移到图内随机位置。放不下时缩到边界内；返回绝对像素坐标。"""
    local = GENERATORS[obj_type](rng)
    xs, ys = local[:, 0], local[:, 1]
    box_w, box_h = w - 2 * margin, h - 2 * margin
    span_x, span_y = xs.max() - xs.min(), ys.max() - ys.min()
    scale = min(box_w / max(span_x, 1.0), box_h / max(span_y, 1.0), 1.0)
    local = local * scale
    xs, ys = local[:, 0], local[:, 1]
    lo_x, hi_x = margin - xs.min(), w - margin - xs.max()
    lo_y, hi_y = margin - ys.min(), h - margin - ys.max()
    if hi_x < lo_x or hi_y < lo_y:  # 浮点边界极端情况：放弃本样本
        return None
    tx = rng.uniform(lo_x, hi_x)
    ty = rng.uniform(lo_y, hi_y)
    return local + np.array([tx, ty])
