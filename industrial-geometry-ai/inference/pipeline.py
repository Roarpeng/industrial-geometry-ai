"""推理 Pipeline（V4 第 51/52/44 节）。

输出 schema（第 52 节）：
{ok, task, objects: [{id, type, center, points: [{u, v, roi_radius, confidence, predicted}]}]}
no-target 门控（第 44 节）：输出点的条件 = 有峰 且 objectness ≥ 0.5。
"""

from __future__ import annotations

import math
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from postprocess.completion import complete_group
from postprocess.grouping import group_points
from postprocess.peak_decode import decode_batch
from task.normalizer import normalize
from task.task_spec import TaskSpec
from task.vocabulary import POINTS_PER_OBJECT, TYPES, TYPE_TO_INDEX


class InferencePipeline:
    def __init__(self, model, device, dcfg: dict, gcfg: dict, W: int = 640, H: int = 360):
        self.model = model.eval().to(device)
        self.device = device
        self.dcfg = dcfg
        self.gcfg = gcfg
        self.W, self.H = W, H

    @torch.no_grad()
    def _forward(self, image: Union[np.ndarray, Image.Image]) -> dict:
        if isinstance(image, Image.Image):
            img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            img = image.astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]
        return self.model(x.to(self.device))

    def run(self, image, spec: Union[TaskSpec, str]) -> dict:
        if isinstance(spec, str):
            spec = TaskSpec.from_target_id(spec)
        t = spec.channel
        tname = TYPES[t]
        k = POINTS_PER_OBJECT[tname]

        outputs = self._forward(image)
        obj_sig = float(outputs["objectness"].sigmoid()[0, t])
        dets = decode_batch(outputs, self.dcfg, self.W, self.H)[0]
        dets = [d for d in dets if d.type_idx == t]

        objects_out = []
        if obj_sig >= 0.5:  # no-target 门控（V4 第 44 节）
            diag_px = math.hypot(self.W, self.H)
            groups, leftovers = group_points(dets, k, self.gcfg, diag_px)
            for i, grp in enumerate(groups):
                extra = complete_group(grp, k, tname)
                pts = grp + ([extra] if extra else [])
                objects_out.append({
                    "id": i, "type": tname,
                    "complete": len(pts) == k,
                    "center": {"u": round(float(np.mean([p.cx for p in pts])) / self.W, 6),
                               "v": round(float(np.mean([p.cy for p in pts])) / self.H, 6)},
                    "points": [{
                        "u": round(p.x / self.W, 6), "v": round(p.y / self.H, 6),
                        "roi_radius": round(p.roi, 6),
                        "confidence": round(p.conf, 4),
                        "predicted": p.predicted,
                    } for p in pts],
                })

        return {"ok": True, "objectness": round(obj_sig, 4),
                "task": spec.to_dict(), "objects": objects_out}

    def run_text(self, image, text: str) -> dict:
        spec = normalize(text)   # 解析失败抛 ClarifyNeeded（V4 第 7.2 节）
        return self.run(image, spec)
