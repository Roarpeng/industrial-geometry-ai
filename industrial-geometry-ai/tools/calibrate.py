"""Confidence 温度校准（V4 第 52 节）。

对 heatmap 峰值 logit 做 temperature scaling：
在验证集上搜索 T 最小化检测点 confidence 与命中（10px 内有 GT）的 NLL。
结果写入 calibration.json（随 checkpoint 发布）。

用法：python -m tools.calibrate --ckpt out/train/best.pth --data data/stagea
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import GeoPointDataset, collate_fn  # noqa: E402
from postprocess.peak_decode import decode_batch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tools.visualize_prediction import load_pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCH_PX = 10.0


@torch.no_grad()
def collect(ckpt_path: str, data_dir: str, max_images: int = 200):
    pipe = load_pipeline(ckpt_path)
    ds = GeoPointDataset([data_dir], tcfg={"gaussian_size_ratio": 0.25, "radius_clamp": [1, 3]},
                         train=False, limit=max_images)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
    logits, hits = [], []
    W, H = pipe.W, pipe.H
    for images, _t, metas in loader:
        outputs = pipe.model(images.to(pipe.device))
        # 峰值处的 logit（decode 用 sigmoid；这里取原始 logit 做校准）
        dets = decode_batch(outputs, pipe.dcfg, W, H)
        logit_map = outputs["heatmap"].cpu()
        import torch.nn.functional as F
        mp = F.max_pool2d(outputs["heatmap"].cpu(), 3, stride=1, padding=1)
        peak_masks = (outputs["heatmap"].cpu() >= mp) & (outputs["heatmap"].cpu().sigmoid() >= 0.05)
        for b, (dets_b, meta) in enumerate(zip(dets, metas)):
            gt = [p for o in meta["objects"] for p in o["points"]]
            for d in dets_b:
                hit = 0
                if any(math.hypot(g[0] - d.x, g[1] - d.y) <= MATCH_PX for g in gt):
                    hit = 1
                # 找到该峰值 cell 的 logit
                cx = int(d.x / 4 / 1)  # 亚像素前的 cell
                cy = int(d.y / 4)
                cy = min(max(cy, 0), peak_masks.shape[2] - 1)
                cx = min(max(cx, 0), peak_masks.shape[3] - 1)
                if bool(peak_masks[b, d.type_idx, cy, cx]):
                    logits.append(float(logit_map[b, d.type_idx, cy, cx]))
                    hits.append(hit)
    return logits, hits


def fit_temperature(logits, hits):
    if not logits:
        return 1.0
    z = torch.tensor(logits)
    y = torch.tensor(hits, dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(z / logT.exp().clamp(0.05, 20.0))
        loss = torch.nn.functional.binary_cross_entropy(
            p.clamp(1e-6, 1 - 1e-6), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "out", "train", "best.pth"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-images", type=int, default=200)
    args = ap.parse_args()

    logits, hits = collect(args.ckpt, args.data, args.max_images)
    acc = sum(hits) / max(len(hits), 1)
    T = fit_temperature(logits, hits)
    out = {"temperature": round(T, 4), "n_points": len(hits), "raw_hit_rate": round(acc, 4)}
    out_path = os.path.join(os.path.dirname(args.ckpt), "calibration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"calibration: {out} -> {out_path}")


if __name__ == "__main__":
    main()
