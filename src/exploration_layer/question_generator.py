"""探索层 · 主动提问生成器。

输入: 未知物体信息 + 场景上下文
输出: 自然语言问题（模板 + 动态填充）
频率限制: 同场景 ≤ 3 次/分钟
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from ..core.types import ObjectPercept

SIZE_WORD = {0.0: "小小", 0.2: "小", 0.4: "中等", 0.6: "较大", 0.8: "很大", 1.0: "很大"}


class QuestionGenerator:
    TEMPLATES = [
        "请问这个{color}的{sizeword}物体是什么？",
        "这个{color}物体属于什么类别？",
        "请问这个物体可以用来做什么？",
        "这个物体是安全的吗？可以触碰吗？",
    ]

    def __init__(self, max_per_minute: int = 3):
        self.max_per_minute = max_per_minute
        self._history: dict[str, deque[float]] = defaultdict(deque)

    def generate(self, obj: ObjectPercept, scene_id: str = "default") -> str | None:
        now = time.monotonic()
        q = self._history[scene_id]
        # 清理窗口外记录
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= self.max_per_minute:
            return None   # 超频，本回合不提问

        color = _color_name(obj.color)
        sizeword = _size_word(obj.size)
        template = self.TEMPLATES[0] if obj.category is None else self.TEMPLATES[3]
        question = template.format(color=color, sizeword=sizeword)
        q.append(now)
        return question

    def reset(self, scene_id: str = "default"):
        self._history[scene_id].clear()


def _color_name(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0.0, min(1.0, c)) for c in rgb)
    if r > 0.6 and g < 0.4 and b < 0.4:
        return "红色"
    if g > 0.6 and r < 0.4 and b < 0.4:
        return "绿色"
    if b > 0.6 and r < 0.4 and g < 0.4:
        return "蓝色"
    if r > 0.6 and g > 0.6 and b < 0.4:
        return "黄色"
    if r < 0.3 and g < 0.3 and b < 0.3:
        return "深色"
    if r > 0.8 and g > 0.8 and b > 0.8:
        return "白色"
    return "彩色"


def _size_word(size: float) -> str:
    for th, w in sorted(SIZE_WORD.items(), reverse=True):
        if size >= th:
            return w
    return "未知大小"
