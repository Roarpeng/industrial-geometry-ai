"""TaskSpec 数据结构（V4 方案第 6 节）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .vocabulary import DEFAULT_OPERATION, TYPES, TYPE_TO_INDEX


@dataclass
class TaskSpec:
    target: str
    operation: str
    scope: str = "all"
    spatial_constraint: Optional[dict] = None
    size_constraint: Optional[dict] = None
    post_operation: Optional[dict] = None

    def __post_init__(self):
        if self.target not in TYPE_TO_INDEX:
            raise ValueError(f"unknown target: {self.target!r}, expected one of {TYPES}")
        if self.scope not in ("all", "single"):
            raise ValueError(f"scope must be 'all' | 'single', got {self.scope!r}")

    @property
    def channel(self) -> int:
        """视觉模型只消费这一个整数（查表，V4 第 6.3 节）。"""
        return TYPE_TO_INDEX[self.target]

    @classmethod
    def from_target_id(cls, target: str) -> "TaskSpec":
        """V0.1 入口：直接给 target_id（V4 第 75 节）。"""
        return cls(target=target, operation=DEFAULT_OPERATION[target], scope="all")

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "operation": self.operation,
            "scope": self.scope,
            "spatial_constraint": self.spatial_constraint,
            "size_constraint": self.size_constraint,
            "post_operation": self.post_operation,
        }
