"""可视化（V4 第 47 节）：GT 绿 / Pred 红 / 补全点黄 / ROI 圆 / 对象编号。"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from PIL import Image, ImageDraw


def draw_gt(image: Image.Image, ann: dict, out_path: str):
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = ann["width"], ann["height"]
    for obj in ann["objects"]:
        pts = [(u * W, v * H) for u, v in obj["points"]]
        for x, y in pts:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(0, 200, 0), width=2)
        x1, y1, x2, y2 = obj["bbox"]
        d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(0, 160, 0), width=1)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        d.text((cx, cy), f"{obj['type']}#{obj['id']}", fill=(0, 160, 0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def draw_prediction(image: Image.Image, result: dict, ann: dict, out_path: str):
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = ann["width"], ann["height"]
    # GT：绿色
    for obj in ann["objects"]:
        if obj["type"] != result["task"]["target"]:
            continue
        for u, v in obj["points"]:
            x, y = u * W, v * H
            d.line([x - 4, y, x + 4, y], fill=(0, 200, 0), width=2)
            d.line([x, y - 4, x, y + 4], fill=(0, 200, 0), width=2)
    # Pred：红（补全点黄）
    for obj in result["objects"]:
        for p in obj["points"]:
            x, y = p["u"] * W, p["v"] * H
            r = p["roi_radius"] * ((W * W + H * H) ** 0.5)
            color = (230, 180, 0) if p["predicted"] else (230, 30, 30)
            d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        c = obj["center"]
        d.text((c["u"] * W, c["v"] * H), f"obj{obj['id']}", fill=(230, 30, 30))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def load_pair(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    img = Image.open(os.path.join(os.path.dirname(json_path), ann["image"]))
    return img, ann
