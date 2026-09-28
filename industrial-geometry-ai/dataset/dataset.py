"""GeoPointDataset：targets 全部 on-the-fly 计算，不落盘（V4 第 26/27/56 节）。

坐标约定（V4 第 10 节）：
- 模型内部 u,v ∈ [0,1]，size/roi 按图像对角线归一化
- heatmap 160×90（stride 4），cell (x,y) 中心像素 = ((x+0.5)·4, (y+0.5)·4)
- offset GT = (center − point) / s（V4 第 20 节，按对象尺寸归一化）
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from task.vocabulary import TYPE_TO_INDEX, TYPES

STRIDE = 4
NUM_TYPES = len(TYPES)


def _gaussian_radius(s_cells: float, min_neighbor_cells: float, cfg: dict) -> int:
    """V4 第 26 节：r = clamp(min(0.25×s_cells, min_neighbor/2), 1, 3)"""
    r = min(cfg["gaussian_size_ratio"] * s_cells, min_neighbor_cells / 2.0)
    lo, hi = cfg["radius_clamp"]
    return int(np.clip(round(r), lo, hi))


def build_targets(objects: List[dict], H: int, W: int, Hc: int, Wc: int, cfg: dict) -> Dict[str, np.ndarray]:
    diag_px = float(np.hypot(W, H))
    diag_cells = float(np.hypot(Hc, Wc))

    heat = np.zeros((NUM_TYPES, Hc, Wc), dtype=np.float32)
    offset = np.zeros((2 * NUM_TYPES, Hc, Wc), dtype=np.float32)
    size_gt = np.zeros((NUM_TYPES, Hc, Wc), dtype=np.float32)
    mask = np.zeros((NUM_TYPES, Hc, Wc), dtype=np.float32)
    objn = np.zeros(NUM_TYPES, dtype=np.float32)

    parsed = []
    for obj in objects:
        t = TYPE_TO_INDEX[obj["type"]]
        pts = np.asarray(obj["points"], dtype=np.float64)
        pts[:, 0] *= W
        pts[:, 1] *= H
        center = pts.mean(axis=0)          # 关键点均值 = 对象中心（四类图形均成立）
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        s = float(np.hypot(x2 - x1, y2 - y1)) / diag_px
        parsed.append(dict(t=t, pts=pts, center=center, s=s))
        objn[t] = 1.0

    type_points = {t: [] for t in range(NUM_TYPES)}
    for p in parsed:
        type_points[p["t"]].extend(map(tuple, p["pts"]))

    for p in parsed:
        t, pts, s = p["t"], p["pts"], p["s"]
        s_px = max(s * diag_px, 1e-6)
        others = np.asarray([q for q in type_points[t]], dtype=np.float64)
        for pt in pts:
            d = np.hypot(others[:, 0] - pt[0], others[:, 1] - pt[1])
            d = d[d > 1e-6]
            min_nbr_cells = (d.min() / STRIDE) if d.size else 1e9
            s_cells = s * diag_cells
            r = _gaussian_radius(s_cells, min_nbr_cells, cfg)
            sigma = (2 * r + 1) / 6.0

            cx = int(np.clip(pt[0] / STRIDE, 0, Wc - 1))
            cy = int(np.clip(pt[1] / STRIDE, 0, Hc - 1))

            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < Hc and 0 <= xx < Wc:
                        val = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
                        heat[t, yy, xx] = max(heat[t, yy, xx], val)
            heat[t, cy, cx] = 1.0
            mask[t, cy, cx] = 1.0
            size_gt[t, cy, cx] = s
            offset[2 * t, cy, cx] = (p["center"][0] - pt[0]) / s_px
            offset[2 * t + 1, cy, cx] = (p["center"][1] - pt[1]) / s_px

    return {"heatmap": heat, "offset": offset, "size": size_gt, "mask": mask, "objectness": objn}


class GeoPointDataset(Dataset):
    """数据目录含 images/*.png 与同名 .json（第 56 节 schema）。"""

    def __init__(self, data_dirs: Sequence[str], tcfg: dict, train: bool = True,
                 limit: int = 0, indices: List[int] = None):
        self.names: List[str] = []
        self.dirs = [d for d in data_dirs if d]
        for d in self.dirs:
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".json") and fn != "manifest.json":
                    self.names.append(os.path.join(d, fn))
        if indices is not None:
            self.names = [self.names[i] for i in indices]
        if limit:
            self.names = self.names[:limit]
        self.tcfg = tcfg
        self.train = train

    def __len__(self):
        return len(self.names)

    def _augment(self, img: np.ndarray, objects: List[dict], rng: np.random.Generator):
        if self.train and rng.random() < 0.5:  # hflip
            img = img[:, ::-1, :].copy()
            H, W = img.shape[:2]
            for obj in objects:
                obj["points"] = [[1.0 - u, v] for u, v in obj["points"]]
                x1, y1, x2, y2 = obj["bbox"]
                obj["bbox"] = [1.0 - x2, y1, 1.0 - x1, y2]
        # photometric
        img = img * rng.uniform(0.85, 1.15)
        img = (img - 0.5) * rng.uniform(0.9, 1.1) + 0.5
        img = np.clip(img, 0, 1)
        img += rng.normal(0, rng.uniform(0.0, 0.02), img.shape)
        return np.clip(img, 0, 1).astype(np.float32)

    def __getitem__(self, idx):
        with open(self.names[idx], "r", encoding="utf-8") as f:
            ann = json.load(f)
        W, H = ann["width"], ann["height"]
        Hc, Wc = H // STRIDE, W // STRIDE

        img_path = os.path.join(os.path.dirname(self.names[idx]), ann["image"])
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

        objects = [dict(type=o["type"], points=[list(p) for p in o["points"]],
                        bbox=list(o["bbox"])) for o in ann["objects"]]
        rng = np.random.default_rng()
        img = self._augment(img, objects, rng)

        targets = build_targets(objects, H, W, Hc, Wc, self.tcfg)

        meta = {"objects": [
            {"type": o["type"], "points": (np.asarray(o["points"]) * [W, H]).tolist()}
            for o in objects]}

        image = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        tgt = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in targets.items()}
        return image, tgt, meta


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    keys = batch[0][1].keys()
    targets = {k: torch.stack([b[1][k] for b in batch]) for k in keys}
    metas = [b[2] for b in batch]
    return images, targets, metas
