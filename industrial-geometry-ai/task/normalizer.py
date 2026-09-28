"""规则版 Instruction Normalizer（V4 方案第 7/8 节）。

第一版不用大模型：同义词归一化 + 槽位抽取。
解析失败必须显式 clarify，禁止静默猜一个 TaskSpec（V4 第 7.2 节）。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .task_spec import TaskSpec
from .vocabulary import DEFAULT_OPERATION, TYPES

# ---- 同义词表（可按现场话术持续扩充）----
TARGET_SYNONYMS = {
    "star": ["五角星", "星形", "五角星形", "星星", "五芒星", "star"],
    "triangle": ["三角形", "三角", "三边形", "triangle"],
    "rectangle": ["矩形", "长方形", "四边形", "方框", "矩形框", "rectangle"],
    "line": ["直线", "线段", "线", "line"],
}

OP_SYNONYMS = {
    "vertices": ["顶点", "角点", "所有角", "各个角", "角", "顶点都", "vertices"],
    "endpoints": ["端点", "两个端点", "起点终点", "两头", "endpoints"],
    "center": ["中心", "圆心", "中心点", "center"],
}

SCOPE_ALL = ["所有", "全部", "每一个", "每个", "各", "都"]
SCOPE_SINGLE = ["这个", "那个", "这条", "该", "单个", "最大", "最小", "指定"]

POST_DIVIDE = re.compile(r"([一二三四五六七八九十\d]+)\s*等分")


class ClarifyNeeded(Exception):
    """解析置信度不足，要求用户重新描述。禁止静默 fallback。"""

    def __init__(self, message: str, candidates: Optional[List[str]] = None):
        super().__init__(message)
        self.candidates = candidates or []


def _find_target(text: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """返回 (target, 候选)。多个命中时交给 clarify。"""
    hits = [t for t in TYPES for syn in TARGET_SYNONYMS[t] if syn in text]
    # 去重并保持 TYPES 顺序
    hits = [t for t in TYPES if t in hits]
    if len(hits) == 1:
        return hits[0], None
    if len(hits) == 0:
        return None, None
    return None, hits


def _find_operation(text: str, target: str) -> str:
    for op, syns in OP_SYNONYMS.items():
        for syn in syns:
            if syn in text:
                return op
    return DEFAULT_OPERATION[target]


def _find_scope(text: str) -> str:
    if any(k in text for k in SCOPE_SINGLE):
        return "single"
    if any(k in text for k in SCOPE_ALL):
        return "all"
    return "all"


def normalize(text: str) -> TaskSpec:
    """自然语言 → TaskSpec；失败抛 ClarifyNeeded。"""
    text = text.strip()
    if not text:
        raise ClarifyNeeded("指令为空，请描述要找的图形，例如：找到所有五角星的顶点")

    target, candidates = _find_target(text)
    if target is None:
        if candidates:
            raise ClarifyNeeded("指令中出现了多种图形，请一次只指定一种", candidates)
        raise ClarifyNeeded(
            "未识别到目标图形，支持的图形：" + "、".join(TYPES), list(TYPES)
        )

    spec = TaskSpec(
        target=target,
        operation=_find_operation(text, target),
        scope=_find_scope(text),
    )

    m = POST_DIVIDE.search(text)
    if m:
        spec.post_operation = {"op": "divide", "n": _cn_num(m.group(1))}

    return spec


def _cn_num(s: str) -> int:
    table = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if s.isdigit():
        return int(s)
    return table.get(s, 0) or 2
