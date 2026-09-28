"""手绘笔触渲染先验。

⚠️ V4 方案第 32 节要求：扰动参数不许拍脑袋。
当前 DEFAULT_PRIOR 是占位值；必须采集 ≥50 段真实手绘笔迹，
实测（弯曲谱 / 断线率 / 线宽分布 / 端点毛刺形态）后写入
dataset/stroke_stats/measured.json 并调用 load_prior() 覆盖。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

_DIR = os.path.dirname(__file__)
_MEASURED = os.path.join(_DIR, "measured.json")


@dataclass
class StrokePrior:
    # 线宽（相对对象尺寸）
    width_base_ratio: float = 0.012   # 线宽 ≈ 对象 bbox 对角线 × 此值
    width_jitter: float = 0.30        # 沿路径 ±30%
    width_min_px: float = 2.0
    width_max_px: float = 7.0
    # 手抖：路径细分后每段的横向扰动（× 线宽）
    tremor_sigma: float = 0.18
    segment_len_px: float = 10.0
    # 断线 / 重复描线 / 不闭合
    break_prob: float = 0.10          # 每段概率
    double_stroke_prob: float = 0.30
    double_stroke_offset_px: float = 1.5
    open_prob: float = 0.15           # 多边形不闭合（缺口占周长比例）
    open_gap_ratio: float = 0.18
    # 端点毛刺
    burr_prob: float = 0.25
    burr_len_px: float = 6.0
    # 墨色
    ink_rgb: tuple = (38, 36, 48)
    ink_color_jitter: float = 12.0


DEFAULT_PRIOR = StrokePrior()


def load_prior() -> StrokePrior:
    """存在实测统计则覆盖默认先验（V4 第 32 节）。"""
    if os.path.exists(_MEASURED):
        with open(_MEASURED, "r", encoding="utf-8") as f:
            return StrokePrior(**json.load(f))
    return DEFAULT_PRIOR


def prior_info() -> dict:
    return {
        "source": "measured" if os.path.exists(_MEASURED) else "placeholder",
        "prior": asdict(DEFAULT_PRIOR),
    }
