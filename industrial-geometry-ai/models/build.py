"""组装完整模型（V4 第 3 节架构图）。"""

from __future__ import annotations

import torch.nn as nn

from .backbone import Backbone
from .fpn import LiteFPN
from .point_head import HeadFusion, PointHeads


class GeoPointModel(nn.Module):
    def __init__(self, mcfg: dict):
        super().__init__()
        chs = tuple(mcfg.get("backbone", {}).get("channels", [48, 96, 192, 384]))
        self.backbone = Backbone(chs)
        self.fpn = LiteFPN(in_dims=chs[1:], out_dim=mcfg.get("fpn", {}).get("lateral_dim", 128))
        hf = mcfg.get("head_fusion", {})
        self.fusion = HeadFusion(p3_dim=mcfg.get("fpn", {}).get("lateral_dim", 128),
                                 skip_in=chs[0], skip_dim=hf.get("skip_dim", 64))
        heads = mcfg.get("heads", {})
        self.heads = PointHeads(hidden=heads.get("hidden_dim", 128),
                                obj_hidden=heads.get("objectness_hidden", 64))

    def forward(self, x):
        feats = self.backbone(x)
        p3 = self.fpn(feats)
        feat = self.fusion(p3, feats["s4"])
        return self.heads(feat)


def build_model(mcfg: dict) -> nn.Module:
    model = GeoPointModel(mcfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model built: {n_params:.2f}M params")
    return model
