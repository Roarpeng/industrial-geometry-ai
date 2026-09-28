"""延迟 benchmark（V4 第 50 节：预算必须实测验证）。

用法：python -m deployment.benchmark --ckpt out/train/best.pth
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

from models.build import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def benchmark(ckpt_path: str | None, H: int = 360, W: int = 640,
              n_warmup: int = 20, n_iter: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(ck["mcfg"])
        model.load_state_dict(ck["ema"])
    else:
        import yaml
        with open(os.path.join(ROOT, "configs", "model.yaml"), encoding="utf-8") as f:
            model = build_model(yaml.safe_load(f))
    model = model.eval().to(device)

    x = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)

    ts.sort()
    print(f"device={device} n={n_iter}")
    print(f"P50={statistics.median(ts):.2f}ms  mean={statistics.mean(ts):.2f}ms  "
          f"min={ts[0]:.2f}ms  max={ts[-1]:.2f}ms")
    print("（模型侧单次前向；Preprocess/Decode/Grouping 另计，见 V4 第 50 节预算表）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "out", "train", "best.pth"))
    args = ap.parse_args()
    benchmark(args.ckpt if os.path.exists(args.ckpt) else None)


if __name__ == "__main__":
    main()
