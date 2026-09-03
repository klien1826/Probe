"""判断层 · 认知安全仲裁（慢系统）。

仲裁规则:
  - 过度提问 (>3次/分钟) → 强制切换至探索
  - 无效循环 (>5次尝试)   → 标记"不可解"，跳过
  - 危险物体 (高压/高温等) → 强制禁止触碰
  - 任务紧急 → 认知探索降级
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.config import Config
from ..core.types import Decision, DecisionPath, ObjectPercept


@dataclass
class ArbitralVerdict:
    allow_ask: bool = True
    allow_touch: bool = True
    force_mode: Optional[str] = None       # 强制探索模式
    blocked: bool = False
    note: str = ""


class CognitiveArbiter:
    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        acfg = cfg.curiosity_of("arbiter")
        self.max_questions_per_min = int(acfg.get("max_questions_per_minute", 3))
        self.max_retry = int(acfg.get("max_retry_cycles", 5))
        forbidden = cfg.safety_of("objects").get("forbidden_categories", [])
        self.forbidden: set[str] = set(forbidden)

    # 状态跟踪
    def _f(self) -> dict:
        if not hasattr(self, "_state"):
            self._state: dict[str, Any] = {
                "question_log": [],        # list[float] timestamps
                "retry_count": {},         # object_id -> int
                "unsolvable": set(),
            }
        return self._state

    def record_ask(self):
        import time
        self._f()["question_log"].append(time.monotonic())

    def record_retry(self, object_id: str):
        st = self._f()
        st["retry_count"][object_id] = st["retry_count"].get(object_id, 0) + 1
        if st["retry_count"][object_id] >= self.max_retry:
            st["unsolvable"].add(object_id)

    def is_unsolvable(self, object_id: str) -> bool:
        return object_id in self._f()["unsolvable"]

    # 主入口
    def arbitrate(
        self,
        decision: Decision,
        obj: Optional[ObjectPercept],
        human: dict[str, Any],
        task: dict[str, Any],
    ) -> ArbitralVerdict:
        import time
        st = self._f()
        verdict = ArbitralVerdict()

        # 1) 危险物体 → 禁止触碰
        if obj is not None:
            if obj.category in self.forbidden or obj.danger_level > 0.8:
                verdict.allow_touch = False
                verdict.note = "dangerous_object_forbidden"

        # 2) 过度提问 → 强制切换至探索
        recent = [t for t in st["question_log"] if time.monotonic() - t <= 60.0]
        if len(recent) >= self.max_questions_per_min and decision.path == DecisionPath.ASK_HUMAN:
            verdict.allow_ask = False
            verdict.force_mode = "broad_scan"
            verdict.note = "over_questioning_force_explore"

        # 3) 无效循环 → 标记不可解，跳过
        if obj is not None and self.is_unsolvable(obj.object_id):
            verdict.blocked = True
            verdict.force_mode = "safe_idle"
            verdict.note = "unsolvable_skip"

        # 4) 任务紧急 → 认知探索降级
        if task.get("urgent", False):
            if verdict.force_mode in (None, "deep_dive"):
                verdict.force_mode = "review_consolidate"
            verdict.note += "; urgent_task_downgrade"

        return verdict
