"""总损失组装（V4 第 28 节）：

L = 1.0×heatmap + 1.0×offset(masked) + 0.3×size(log, masked) + 0.2×objectness
"""

from __future__ import annotations

from typing import Dict

import torch

from .aux_loss import objectness_loss, offset_loss, size_log_loss
from .heatmap_loss import heatmap_focal_loss


def compute_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                 weights: Dict[str, float] = None, focal: Dict[str, float] = None) \
        -> Dict[str, torch.Tensor]:
    weights = weights or {"heatmap": 1.0, "offset": 1.0, "size": 0.3, "objectness": 0.2}
    focal = focal or {"alpha": 2.0, "beta": 4.0}

    l_heat = heatmap_focal_loss(outputs["heatmap"], targets["heatmap"],
                                alpha=focal["alpha"], beta=focal["beta"])
    l_off = offset_loss(outputs["offset"], targets["offset"], targets["mask"])
    l_size = size_log_loss(outputs["size"], targets["size"], targets["mask"])
    l_obj = objectness_loss(outputs["objectness"], targets["objectness"])

    total = (weights["heatmap"] * l_heat + weights["offset"] * l_off
             + weights["size"] * l_size + weights["objectness"] * l_obj)
    return {"total": total, "heatmap": l_heat.detach(),
            "offset": l_off.detach(), "size": l_size.detach(), "objectness": l_obj.detach()}
