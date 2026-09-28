"""合成数据生成器 CLI（V4 方案第 30/31/56 节）。

输出：PNG + 同名 JSON（第 56 节 schema）+ manifest.json。
硬约束（第 31.1 节）：同 type 关键点间距 ≥8px、跨 type ≥4px、
bbox IoA ≤30%、全部关键点落在 border_margin 内；不满足则重采样。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from dataset.stroke_stats.prior import load_prior
from . import render
from .shapes import gen_object

TYPES = ("triangle", "rectangle", "star", "line")


def _bbox(pts: np.ndarray):
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def _ioa(b1, b2) -> float:
    ix = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = ix * iy
    a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / min(a1, a2)


def _valid(new_pts, new_type, placed, cfg) -> bool:
    w, h = cfg["w"], cfg["h"]
    m = cfg["margin"]
    if (new_pts[:, 0] < m).any() or (new_pts[:, 0] > w - m).any() \
       or (new_pts[:, 1] < m).any() or (new_pts[:, 1] > h - m).any():
        return False
    # 对象自身内部点间距（V4 31.1：同 type 任意两关键点，含同对象）
    dd_self = np.hypot(new_pts[:, None, 0] - new_pts[None, :, 0],
                       new_pts[:, None, 1] - new_pts[None, :, 1])
    np.fill_diagonal(dd_self, np.inf)
    if dd_self.min() < cfg["min_spacing"]:
        return False
    nb = _bbox(new_pts)
    for t, pts, bb in placed:
        d_min = cfg["min_spacing"] if t == new_type else cfg["cross_spacing"]
        dd = np.hypot(pts[:, None, 0] - new_pts[None, :, 0],
                      pts[:, None, 1] - new_pts[None, :, 1])
        if dd.min() < d_min:
            return False
        if _ioa(nb, bb) > cfg["max_overlap"]:
            return False
    return True


def generate_image(rng: np.random.Generator, cfg: dict):
    n_target = rng.integers(cfg["objects_min"], cfg["objects_max"] + 1)
    if rng.random() < cfg["negative_ratio"]:
        n_target = 0

    img = render.make_background(rng, cfg["w"], cfg["h"], cfg["mode"])
    draw = ImageDraw.Draw(img)
    prior = load_prior()
    diag = float(np.hypot(cfg["w"], cfg["h"]))

    placed = []
    objects = []
    tries = 0
    while len(placed) < n_target and tries < n_target * 60:
        tries += 1
        types = cfg.get("types", TYPES)
        t = types[int(rng.integers(0, len(types)))]
        pts = gen_object(rng, t, cfg["w"], cfg["h"], cfg["margin"])
        if pts is None or not _valid(pts, t, placed, cfg):
            continue
        bb = _bbox(pts)
        placed.append((t, pts, bb))
        obj_diag = float(np.hypot(bb[2] - bb[0], bb[3] - bb[1]))
        if t == "line":
            render.draw_curved_line(draw, tuple(pts[0]), tuple(pts[1]), rng, prior)
        else:
            render.draw_stroke(draw, pts, closed=True, rng=rng, prior=prior, diag_px=obj_diag)
        objects.append({
            "id": len(objects),
            "type": t,
            "points": [[round(float(x) / cfg["w"], 6), round(float(y) / cfg["h"], 6)]
                       for x, y in pts],
            "bbox": [round(v, 6) for v in
                     (bb[0] / cfg["w"], bb[1] / cfg["h"], bb[2] / cfg["w"], bb[3] / cfg["h"])],
        })

    ann = {
        "image": None,
        "width": cfg["w"],
        "height": cfg["h"],
        "objects": objects,
        "source": "synthetic",
    }
    return img, ann


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=5000)
    ap.add_argument("--out", type=str, default="data/synthetic")
    ap.add_argument("--mode", type=str, default="clean", choices=["clean", "industrial"])
    ap.add_argument("--objects-min", type=int, default=1)
    ap.add_argument("--objects-max", type=int, default=5)
    ap.add_argument("--negative-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dense", action="store_true", help="压力集：对象数 20~60")
    ap.add_argument("--types", type=str, default="triangle,rectangle,star,line",
                    help="逗号分隔的类型子集（弱类偏采样用）")
    args = ap.parse_args(argv)

    types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    cfg = {
        "w": 640, "h": 360, "mode": args.mode,
        "types": types,
        "objects_min": 20 if args.dense else args.objects_min,
        "objects_max": 60 if args.dense else args.objects_max,
        "negative_ratio": args.negative_ratio,
        "min_spacing": 8.0, "cross_spacing": 4.0,
        "max_overlap": 0.30, "margin": 24.0,
    }
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = []
    for i in range(args.num):
        img, ann = generate_image(rng, cfg)
        name = f"{args.mode}_{i:06d}"
        img.save(os.path.join(args.out, "images", name + ".png"))
        ann["image"] = f"images/{name}.png"
        with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False)
        files.append(name + ".json")
        if (i + 1) % 500 == 0:
            print(f"generated {i + 1}/{args.num}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "num": args.num, "seed": args.seed, "files": files},
                  f, ensure_ascii=False)
    print(f"done: {args.num} images -> {args.out}")


if __name__ == "__main__":
    main()
