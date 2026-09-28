"""渲染层：工业背景（含 hard negative 干扰物）+ 手绘笔触（V4 第 33/34 节）。"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageDraw

from dataset.stroke_stats.prior import StrokePrior

Point = tuple  # (x, y)


# ---------------------------------------------------------------- 背景

def _base_clean(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    base = 232.0 + rng.uniform(-8, 8)
    grad = np.linspace(0, 1, h, dtype=np.float32)[:, None] * rng.uniform(-6, 6)
    img = np.full((h, w, 3), base, dtype=np.float32) + grad[:, :, None]
    img += rng.normal(0, 3.5, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _base_wood(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    tan = np.array([196.0, 158.0, 110.0])
    angle = rng.uniform(0, math.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx * math.cos(angle) + yy * math.sin(angle)
    stripes = 6.0 * np.sin(u / rng.uniform(9, 18) + rng.uniform(0, 10))
    stripes += 3.0 * np.sin(u / rng.uniform(40, 90) + rng.uniform(0, 10))
    img = tan[None, None, :] + stripes[:, :, None]
    img += rng.normal(0, 5.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _base_metal(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    base = np.array([148.0, 152.0, 158.0])
    img = np.tile(base[None, None, :], (h, w, 1)).astype(np.float32)
    img += rng.normal(0, 4.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _distractors(draw: ImageDraw.ImageDraw, rng: np.random.Generator, w: int, h: int) -> None:
    """hard negative：看起来像目标但不是目标（V4 第 34 节）。"""
    n = int(rng.integers(4, 12))
    for _ in range(n):
        kind = rng.integers(0, 4)
        if kind == 0:  # 螺丝 → 假点
            r = rng.uniform(3, 7)
            cx, cy = rng.uniform(r, w - r), rng.uniform(r, h - r)
            ang = rng.uniform(0, math.pi)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(70, 70, 74), width=2)
            draw.line([cx - r * math.cos(ang), cy - r * math.sin(ang),
                       cx + r * math.cos(ang), cy + r * math.sin(ang)], fill=(70, 70, 74), width=2)
        elif kind == 1:  # 划痕/木纹缝 → 假线
            x1, y1 = rng.uniform(0, w), rng.uniform(0, h)
            ln, ang = rng.uniform(40, 220), rng.uniform(0, math.pi)
            x2, y2 = x1 + ln * math.cos(ang), y1 + ln * math.sin(ang)
            draw.line([x1, y1, x2, y2], fill=(120, 112, 100), width=1)
        elif kind == 2:  # 安装孔 → 假圆点
            r = rng.uniform(4, 10)
            cx, cy = rng.uniform(r, w - r), rng.uniform(r, h - r)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 58, 62))
        else:  # 夹具边缘 → 假直线
            x1, y1 = rng.uniform(0, w), rng.uniform(0, h)
            ln, ang = rng.uniform(80, 300), rng.choice([0, math.pi / 2])
            x2, y2 = x1 + ln * math.cos(ang), y1 + ln * math.sin(ang)
            draw.line([x1, y1, x2, y2], fill=(96, 96, 100), width=3)


def make_background(rng: np.random.Generator, w: int, h: int, mode: str) -> Image.Image:
    if mode == "clean":
        arr = _base_clean(rng, w, h)
        img = Image.fromarray(arr)
        if rng.random() < 0.15:  # 少量假点干扰
            d = ImageDraw.Draw(img)
            _distractors(d, rng, w, h)
        return img
    # industrial
    mat = rng.choice(["wood", "metal", "clean"])
    if mat == "wood":
        arr = _base_wood(rng, w, h)
    elif mat == "metal":
        arr = _base_metal(rng, w, h)
    else:
        arr = _base_clean(rng, w, h)
    img = Image.fromarray(arr)
    _distractors(ImageDraw.Draw(img), rng, w, h)
    return img


# ---------------------------------------------------------------- 笔触

def _path_with_tremor(p1, p2, rng, prior: StrokePrior) -> List[Point]:
    n = max(2, int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / prior.segment_len_px))
    n = min(n, 40)
    nx, ny = -(p2[1] - p1[1]), (p2[0] - p1[0])
    ln = math.hypot(nx, ny) + 1e-6
    nx, ny = nx / ln, ny / ln
    pts = []
    for i in range(n + 1):
        t = i / n
        off = rng.normal(0, prior.tremor_sigma * 4.0)  # px
        pts.append((p1[0] + (p2[0] - p1[0]) * t + nx * off,
                    p1[1] + (p2[1] - p1[1]) * t + ny * off))
    return pts


def _draw_polyline(draw, pts: Sequence[Point], width: float, fill) -> None:
    if len(pts) < 2:
        return
    seg = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    for a, b in seg:
        draw.line([a, b], fill=fill, width=max(1, int(round(width))), joint="curve")


def draw_stroke(draw: ImageDraw.ImageDraw, keypoints: np.ndarray, closed: bool,
                rng: np.random.Generator, prior: StrokePrior, diag_px: float) -> None:
    """沿关键点骨架渲染手绘笔画。"""
    width = float(np.clip(prior.width_base_ratio * diag_px,
                          prior.width_min_px, prior.width_max_px))
    ink = tuple(int(np.clip(c + rng.normal(0, prior.ink_color_jitter), 0, 255))
                for c in prior.ink_rgb)

    pts_all = [np.asarray(keypoints[0], dtype=np.float64)]
    edges = list(range(len(keypoints) - 1))
    if closed:
        edges.append(len(keypoints) - 1)  # 回到起点

    n_open = len(edges)
    skip_edge = -1
    if closed and rng.random() < prior.open_prob:
        skip_edge = int(rng.integers(0, n_open))  # 不闭合：跳过一条边

    for e in edges:
        if e == skip_edge:
            continue
        a = keypoints[e]
        b = keypoints[(e + 1) % len(keypoints)] if closed else keypoints[e + 1]
        path = _path_with_tremor(tuple(a), tuple(b), rng, prior)
        if rng.random() < prior.break_prob and len(path) > 4:  # 断线
            cut = int(len(path) * rng.uniform(0.3, 0.7))
            path = path[:cut]
        for i in range(len(path) - 1):
            w_seg = width * rng.uniform(1 - prior.width_jitter, 1 + prior.width_jitter)
            _draw_polyline(draw, path[i:i + 2], w_seg, ink)
        if rng.random() < prior.double_stroke_prob:  # 重复描线
            off = prior.double_stroke_offset_px
            path2 = [(x + rng.normal(0, off), y + rng.normal(0, off)) for (x, y) in path]
            _draw_polyline(draw, path2, width * 0.8, ink)

    if rng.random() < prior.burr_prob:  # 端点毛刺
        tip = keypoints[0]
        ang = rng.uniform(0, 2 * math.pi)
        ln = prior.burr_len_px * rng.uniform(0.5, 1.2)
        draw.line([tuple(tip), (tip[0] + ln * math.cos(ang), tip[1] + ln * math.sin(ang))],
                  fill=ink, width=max(1, int(width * 0.7)))


def draw_curved_line(draw: ImageDraw.ImageDraw, p1, p2,
                      rng: np.random.Generator, prior: StrokePrior) -> None:
    """直线对象：带曲率的路径（关键点仍是两个端点）。"""
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    sag = length * rng.uniform(0.0, 0.08)
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    nx, ny = -(p2[1] - p1[1]) / (length + 1e-6), (p2[0] - p1[0]) / (length + 1e-6)
    ctrl = (mx + nx * sag * rng.uniform(-1, 1), my + ny * sag)
    # 二次贝塞尔细分
    pts = []
    for i in range(24):
        t = i / 23
        x = (1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p2[1]
        pts.append((x, y))
    width = float(np.clip(prior.width_base_ratio * length * 2.0,
                          prior.width_min_px, prior.width_max_px))
    ink = tuple(int(np.clip(c + rng.normal(0, prior.ink_color_jitter), 0, 255))
                for c in prior.ink_rgb)
    for i in range(len(pts) - 1):
        w_seg = width * rng.uniform(1 - prior.width_jitter, 1 + prior.width_jitter)
        _draw_polyline(draw, pts[i:i + 2], w_seg, ink)
