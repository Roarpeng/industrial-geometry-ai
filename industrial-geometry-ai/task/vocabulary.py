"""target → channel 固定查表（V4 方案第 6.3 节）。

视觉模型内部不存在任何语言/条件逻辑：
TaskSpec.target 只在这里被翻译成通道索引。
新增图形类型 = 在 TYPES 里追加一项 + 更新 heads 通道数。
"""

from __future__ import annotations

from typing import Dict, Tuple

# 顺序即通道顺序，写死后存入 checkpoint（task_vocabulary.json）
TYPES: Tuple[str, ...] = ("triangle", "rectangle", "star", "line")
TYPE_TO_INDEX: Dict[str, int] = {t: i for i, t in enumerate(TYPES)}

# 每类对象的关键点数量 K（分组与校验的强先验，V4 第 1/22 节）
POINTS_PER_OBJECT: Dict[str, int] = {
    "triangle": 3,    # 顶点
    "rectangle": 4,   # 角点
    "star": 10,       # 5 外顶点 + 5 内顶点
    "line": 2,        # 端点
}

# target 缺省 operation（V4 第 7.2 节兜底表）
DEFAULT_OPERATION: Dict[str, str] = {
    "triangle": "vertices",
    "rectangle": "vertices",
    "star": "vertices",
    "line": "endpoints",
}

# 通道布局：heatmap/size 每 type 1ch，offset 每 type 2ch
def heatmap_channel(target: str) -> int:
    return TYPE_TO_INDEX[target]


def offset_channels(target: str) -> Tuple[int, int]:
    i = TYPE_TO_INDEX[target]
    return 2 * i, 2 * i + 1


def size_channel(target: str) -> int:
    return TYPE_TO_INDEX[target]


def vocabulary_dict() -> dict:
    return {
        "types": list(TYPES),
        "type_to_index": TYPE_TO_INDEX,
        "points_per_object": POINTS_PER_OBJECT,
        "default_operation": DEFAULT_OPERATION,
    }
