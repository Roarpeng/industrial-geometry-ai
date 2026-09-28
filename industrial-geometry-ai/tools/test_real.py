"""真实采集图测试：letterbox 推理 → 坐标/ROI 映射回原图 → 原分辨率上画标记。

用法：
    python tools/test_real.py --ckpt out/phase0/best.pth \
        --images ../test_imgs --out out/real_test
对每张图跑全部 4 类，输出：<name>_annotated.jpg（红=检测点+ROI，黄=几何补全点）。
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task.vocabulary import TYPES  # noqa: E402
from tools.self_train import letterbox, unmap  # noqa: E402
from tools.visualize_prediction import load_pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = {
    "triangle": (230, 30, 30),
    "rectangle": (30, 90, 230),
    "star": (230, 120, 20),
    "line": (20, 160, 60),
}


def annotate(pipe, img_path: str, out_path: str, conf: float = 0.10):
    raw = Image.open(img_path).convert("RGB")
    W0, H0 = raw.size
    diag0 = math.hypot(W0, H0)
    canvas, scale, ox, oy = letterbox(raw, pipe.W, pipe.H)

    draw = ImageDraw.Draw(raw)
    summary = []
    for tname in TYPES:
        result = pipe.run(canvas, tname)
        for obj in result["objects"]:
            color = COLORS[tname]
            for p in obj["points"]:
                x, y = unmap(p["u"], p["v"], scale, ox, oy, W0, H0)
                r = p["roi_radius"] * diag0
                col = (230, 190, 0) if p["predicted"] else color
                draw.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3)
                draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col)
                if p["predicted"]:
                    draw.line([x - 7, y, x + 7, y], fill=col, width=2)
                    draw.line([x, y - 7, x, y + 7], fill=col, width=2)
            c = obj["center"]
            cx, cy = unmap(c["u"], c["v"], scale, ox, oy, W0, H0)
            draw.text((cx - 14, cy - 8), f"{tname}#{obj['id']}", fill=color)
            n_pred = sum(1 for p in obj["points"] if p["predicted"])
            summary.append(f"{tname}#{obj['id']}:{len(obj['points'])}pts"
                           + (f"(+{n_pred}补全)" if n_pred else ""))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    raw.save(out_path)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "out", "phase0", "best.pth"))
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "real_test"))
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=None,
                    help="覆盖 decode 峰值阈值（默认用 decode_params 里的 0.10）")
    args = ap.parse_args()

    pipe = load_pipeline(args.ckpt)
    if args.tau is not None:
        pipe.dcfg["peak_threshold"] = args.tau
    files = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    for fn in files:
        stem, ext = os.path.splitext(fn)
        out = os.path.join(args.out, f"{stem}_{ext.lstrip('.')}_annotated.jpg")
        summary = annotate(pipe, os.path.join(args.images, fn), out, args.conf)
        print(f"{fn}: {' '.join(summary) if summary else '(无检测)'}")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
