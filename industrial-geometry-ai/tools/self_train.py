"""Self-training 伪标注工具（V4 方案第 39.2 节）。

流程：
    未标注真实图（任意尺寸，letterbox 到 640×360）
     → 模型伪标注（decode + 分组 + 补全）
     → confidence 过滤（只保留"完整对象且全部点置信度达标"的）
     → 输出第 56 节 schema 的 JSON（source=pseudo，坐标映射回原图）
     → 随机抽 10% 生成 spotcheck 清单供人工复核

用法：
    python tools/self_train.py --ckpt out/train/best.pth --images data/real_raw --out data/pseudo
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.pipeline import InferencePipeline  # noqa: E402
from task.vocabulary import POINTS_PER_OBJECT, TYPES  # noqa: E402
from tools.visualize_prediction import load_pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def letterbox(img: Image.Image, W: int = 640, H: int = 360):
    """等比缩放 + 中性灰 padding（V4 第 10.1 节：禁止非等比 resize）。"""
    w, h = img.size
    scale = min(W / w, H / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (W, H), (128, 128, 128))
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy


def unmap(u: float, v: float, scale: float, ox: int, oy: int, w: int, h: int):
    """模型输出 (u,v) → 原图像素坐标。"""
    x = (u * 640 - ox) / scale
    y = (v * 360 - oy) / scale
    return min(max(x, 0), w - 1), min(max(y, 0), h - 1)


@torch.no_grad()
def pseudo_label(pipe: InferencePipeline, img_path: str, conf_thresh: float) -> tuple:
    """返回 (第 56 节 schema 的标注 dict, 全图最高点置信度)。"""
    raw = Image.open(img_path)
    w, h = raw.size
    canvas, scale, ox, oy = letterbox(raw, pipe.W, pipe.H)

    objects = []
    max_conf = 0.0
    for tname in TYPES:
        result = pipe.run(canvas, tname)
        for obj in result["objects"]:
            pts = obj["points"]
            max_conf = max(max_conf, max(p["confidence"] for p in pts))
            if not obj["complete"]:
                continue
            if any(p["confidence"] < conf_thresh for p in pts):
                continue  # 只收高置信完整对象（V4 第 39.2 节过滤）
            xy = [unmap(p["u"], p["v"], scale, ox, oy, w, h) for p in pts]
            xs = [p[0] for p in xy]; ys = [p[1] for p in xy]
            objects.append({
                "id": len(objects),
                "type": tname,
                "points": [[round(x / w, 6), round(y / h, 6)] for x, y in xy],
                "bbox": [round(min(xs) / w, 6), round(min(ys) / h, 6),
                         round(max(xs) / w, 6), round(max(ys) / h, 6)],
            })
    ann = {"image": None, "width": w, "height": h,
           "objects": objects, "source": "pseudo"}
    return ann, max_conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "out", "train", "best.pth"))
    ap.add_argument("--images", required=True, help="未标注真实图目录")
    ap.add_argument("--out", required=True, help="输出目录（JSON）")
    ap.add_argument("--conf-thresh", type=float, default=0.6)
    ap.add_argument("--spotcheck-ratio", type=float, default=0.10)
    ap.add_argument("--min-conf-label", type=float, default=0.0,
                    help="图内最高置信度低于此值 → 记录为疑似 negative-only 而不是强标注")
    args = ap.parse_args()

    pipe = load_pipeline(args.ckpt)
    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    if not files:
        print(f"no images in {args.images}")
        return

    rng = random.Random(0)
    spotcheck, n_obj_total, n_suspect_neg = [], 0, 0
    for i, fn in enumerate(files):
        ann, max_conf = pseudo_label(pipe, os.path.join(args.images, fn), args.conf_thresh)
        name = os.path.splitext(fn)[0]
        if not ann["objects"] and max_conf >= args.min_conf_label:
            n_suspect_neg += 1
        ann["image"] = fn
        ann["meta"] = {"max_conf_any": round(max_conf, 4)}
        with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False)
        n_obj_total += len(ann["objects"])
        if rng.random() < args.spotcheck_ratio:
            spotcheck.append(name)

    with open(os.path.join(args.out, "spotcheck.json"), "w", encoding="utf-8") as f:
        json.dump({"ratio": args.spotcheck_ratio, "files": spotcheck,
                   "note": "人工复核这些伪标注后再入训练集（V4 第 39.2 节）"},
                  f, ensure_ascii=False, indent=2)
    print(f"pseudo-labeled {len(files)} images, {n_obj_total} objects, "
          f"{n_suspect_neg} suspected negative-only, spotcheck {len(spotcheck)} files")
    print(f"-> {args.out}（含 spotcheck.json）")


if __name__ == "__main__":
    main()
