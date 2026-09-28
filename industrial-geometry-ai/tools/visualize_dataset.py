"""数据集可视化 CLI：python tools/visualize_dataset.py --data data/synthetic --num 8"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.visualize import draw_gt, load_pair  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--out", default="out/dataset_vis")
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.data)
                   if f.endswith(".json") and f != "manifest.json")[:args.num]
    for fn in names:
        img, ann = load_pair(os.path.join(args.data, fn))
        out = os.path.join(args.out, fn.replace(".json", ".png"))
        draw_gt(img, ann, out)
        print("saved", out)


if __name__ == "__main__":
    main()
