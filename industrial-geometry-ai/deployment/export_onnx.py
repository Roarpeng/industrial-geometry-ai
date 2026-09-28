"""ONNX 导出（V4 第 46/58 节）。

用法：python -m deployment.export_onnx --ckpt out/train/best.pth --out out/model.onnx
"""

from __future__ import annotations

import argparse
import os

import torch

from models.build import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export(ckpt_path: str, out_path: str, H: int = 360, W: int = 640, opset: int = 17):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["mcfg"])
    model.load_state_dict(ck["ema"])
    model.eval()

    dummy = torch.randn(1, 3, H, W)
    output_names = ["heatmap", "offset", "size", "objectness"]
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["image"],
        output_names=output_names,
        opset_version=opset,
        dynamic_axes={"image": {0: "batch"}},
        dynamo=False,  # 传统 TorchScript exporter，避免依赖 onnxscript
    )
    print(f"exported: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "out", "train", "best.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "model.onnx"))
    args = ap.parse_args()
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
