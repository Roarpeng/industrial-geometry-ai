"""预测可视化 CLI：python tools/visualize_prediction.py --ckpt out/train/best.pth --data data/synthetic"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import yaml  # noqa: E402

from inference.pipeline import InferencePipeline  # noqa: E402
from models.build import build_model  # noqa: E402
from task.vocabulary import TYPES  # noqa: E402
from tools.visualize import draw_prediction, load_pair  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pipeline(ckpt_path: str, device=None) -> InferencePipeline:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["mcfg"])
    model.load_state_dict(ck["ema"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(ROOT, "configs", "dataset.yaml"), encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)
    decode_cfg = {**dcfg["decode"], "roi": dcfg["roi"]}
    return InferencePipeline(model, torch.device(device), decode_cfg, dcfg["grouping"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "out", "train", "best.pth"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--target", default=None, help="默认轮流 4 类")
    ap.add_argument("--out", default="out/pred_vis")
    args = ap.parse_args()

    pipe = load_pipeline(args.ckpt)
    names = sorted(f for f in os.listdir(args.data)
                   if f.endswith(".json") and f != "manifest.json")[:args.num]
    for i, fn in enumerate(names):
        img, ann = load_pair(os.path.join(args.data, fn))
        target = args.target or TYPES[i % len(TYPES)]
        result = pipe.run(img, target)
        out = os.path.join(args.out, f"{fn.replace('.json', '')}_{target}.png")
        draw_prediction(img, result, ann, out)
        print(f"saved {out} target={target} objects={len(result['objects'])}")


if __name__ == "__main__":
    main()
