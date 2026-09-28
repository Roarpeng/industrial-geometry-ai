"""解码阈值扫描（V4 第 21 节 τ=0.30 为初值，需数据验证后固定）。

对验证集扫描 peak_threshold，报告 V0.1 三指标（point_recall / roi_hit /
no_target）与综合分（三者最小值），给出推荐 τ。

用法：python tools/sweep_thresholds.py --ckpt out/phase0/best.pth \
          --data data/stagea,data/stageb --taus 0.20 0.25 0.30 0.35 0.40
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import GeoPointDataset, collate_fn  # noqa: E402
from models.build import build_model  # noqa: E402
from training.metrics import evaluate  # noqa: E402
from training.train import _split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.20, 0.25, 0.30, 0.35, 0.40])
    ap.add_argument("--val-limit", type=int, default=250)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["mcfg"])
    model.load_state_dict(ck["ema"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    with open(os.path.join(ROOT, "configs", "dataset.yaml"), encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)

    dirs = [d if os.path.isabs(d) else os.path.join(ROOT, d)
            for d in args.data.split(",")]
    base = GeoPointDataset(dirs, tcfg=dcfg["target"], train=False)
    _, val_idx = _split(base.names)
    if args.val_limit:
        val_idx = val_idx[:args.val_limit]
    val_ds = GeoPointDataset(dirs, tcfg=dcfg["target"], train=False, indices=val_idx)
    loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn, num_workers=4)
    print(f"val images: {len(val_ds)}  device: {device}")

    W, H = 640, 360
    rows = []
    for tau in args.taus:
        decode_cfg = {**dcfg["decode"], "roi": dcfg["roi"], "peak_threshold": tau}
        m = evaluate(model, loader, device, decode_cfg, dcfg["grouping"], W, H)
        score = min(m["point_recall"], m["roi_hit"], m["no_target"])
        rows.append((tau, m["point_recall"], m["roi_hit"], m["no_target"],
                     m["object_recall"], score))
        print(f"τ={tau:.2f}  point={m['point_recall']:.4f}  roi={m['roi_hit']:.4f}  "
              f"no_tgt={m['no_target']:.4f}  obj={m['object_recall']:.4f}  min={score:.4f}")

    best = max(rows, key=lambda r: r[-1])
    print(f"\n推荐 τ={best[0]:.2f}（V0.1 三指标最小值 {best[-1]:.4f}）")
    if best[-1] < 0.95:
        print("仍未达 V0.1 线（0.95）：进入下一轮迭代（续训 / 加数据）")


if __name__ == "__main__":
    main()
