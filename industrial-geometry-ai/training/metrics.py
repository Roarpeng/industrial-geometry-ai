"""验证指标（V4 第 49 节）：

- Point Recall：固定 10px 阈值（与 ROI 解耦）
- ROI Hit：dist ≤ roi_pred 且 roi_pred ≤ max(0.3×s_gt, min_r)（带上界约束）
- mean(roi_pred / s_gt)：防止整体报大
- Object Recall：对象全部 K 点命中（允许补全点）
- No-target：无该 type 时正确输出空（objectness 门控，V4 第 44 节）
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from postprocess.peak_decode import decode_batch
from task.vocabulary import TYPES

POINT_MATCH_PX = 10.0
ROI_HIT_RATIO = 0.3


def _new_stats():
    return {"gt_pts": 0, "matched": 0, "roi_pairs": 0, "roi_hits": 0, "roi_ratio_sum": 0.0,
            "gt_obj": 0, "obj_complete": 0, "no_total": 0, "no_correct": 0}


@torch.no_grad()
def evaluate(model, loader, device, dcfg: dict, gcfg: dict, W: int, H: int) -> Dict[str, float]:
    model.eval()
    stats = {t: _new_stats() for t in TYPES}
    min_r = dcfg.get("roi", {}).get("min_r", 0.01)

    for images, _targets, metas in loader:
        images = images.to(device)
        outputs = model(images)
        obj_sig = outputs["objectness"].sigmoid().cpu()
        dets_all = decode_batch(outputs, dcfg, W, H)

        for b, (dets, meta) in enumerate(zip(dets_all, metas)):
            for ti, tname in enumerate(TYPES):
                st = stats[tname]
                preds = [d for d in dets if d.type_idx == ti]
                if float(obj_sig[b, ti]) < 0.5:  # no-target 门控（V4 第 44 节）
                    preds = []

                # GT 点（带所属对象与 s_gt）
                gt_entries = []
                gt_obj_matched: Dict[int, int] = {}
                gt_obj_total: Dict[int, int] = {}
                for oi, obj in enumerate(meta["objects"]):
                    if obj["type"] != tname:
                        continue
                    pts = obj["points"]
                    x1 = min(p[0] for p in pts); y1 = min(p[1] for p in pts)
                    x2 = max(p[0] for p in pts); y2 = max(p[1] for p in pts)
                    s_gt = math.hypot(x2 - x1, y2 - y1) / math.hypot(W, H)
                    gt_obj_total[oi] = len(pts)
                    gt_obj_matched.setdefault(oi, 0)
                    for p in pts:
                        gt_entries.append((p[0], p[1], s_gt, oi))

                if not gt_entries:
                    st["no_total"] += 1
                    if len(preds) == 0:
                        st["no_correct"] += 1
                    continue
                st["gt_obj"] += len(gt_obj_total)

                # 贪心一对一匹配（≤10px）
                pairs = []
                for gi, g in enumerate(gt_entries):
                    for pj, p in enumerate(preds):
                        d = math.hypot(g[0] - p.x, g[1] - p.y)
                        if d <= POINT_MATCH_PX:
                            pairs.append((d, gi, pj))
                pairs.sort(key=lambda x: x[0])
                used_g, used_p = set(), set()
                match: List = []
                for d, gi, pj in pairs:
                    if gi in used_g or pj in used_p:
                        continue
                    used_g.add(gi); used_p.add(pj)
                    match.append((gi, pj, d))

                st["gt_pts"] += len(gt_entries)
                st["matched"] += len(match)

                for gi, pj, d in match:
                    g = gt_entries[gi]
                    p = preds[pj]
                    st["roi_pairs"] += 1
                    if d <= p.roi * math.hypot(W, H) and p.roi <= max(ROI_HIT_RATIO * g[2], min_r):
                        st["roi_hits"] += 1
                    st["roi_ratio_sum"] += p.roi / max(g[2], 1e-6)
                    gt_obj_matched[g[3]] += 1

                st["obj_complete"] += sum(1 for oi in gt_obj_total
                                          if gt_obj_matched[oi] == gt_obj_total[oi])

    def _safe(n, d):
        return n / d if d else float("nan")

    result = {}
    for tname in TYPES:
        st = stats[tname]
        result.update({
            f"point_recall/{tname}": _safe(st["matched"], st["gt_pts"]),
            f"roi_hit/{tname}": _safe(st["roi_hits"], st["roi_pairs"]),
            f"object_recall/{tname}": _safe(st["obj_complete"], st["gt_obj"]),
            f"no_target/{tname}": _safe(st["no_correct"], st["no_total"]),
        })
        if st["roi_pairs"]:
            result[f"roi_ratio/{tname}"] = st["roi_ratio_sum"] / st["roi_pairs"]
    for key in ("point_recall", "roi_hit", "object_recall", "no_target"):
        vals = [v for k, v in result.items() if k.startswith(key + "/")
                and v == v]  # 过滤 nan
        result[key] = sum(vals) / len(vals) if vals else float("nan")
    return result
