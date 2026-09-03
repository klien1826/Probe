"""探索层 · 双路径决策引擎（规则评分表）。

评分维度（总分 0-10，得分越高越倾向"问人"）:
  - 风险等级   (权重 0.25): 物体越危险 → 越应问人      → score = danger*10
  - 人类空闲度 (权重 0.20): 人在且空闲 → 问人成本低     → present&idle=10, present&busy=5, absent=2
  - 探索成本   (权重 0.20): 自主探索越昂贵 → 越应问人   → score = cost*10
  - 任务优先级 (权重 0.20): 任务越紧急 → 越不应停下问人 → score = (1-urgency)*10
  - 物体动态性 (权重 0.15): 物体静止 → 问人信息不过时   → score = (1-dynamicity)*10

决策: 总分 >= 6.0 → 问人; < 6.0 → 自主探索
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.config import Config
from ..core.types import Decision, DecisionPath, ObjectPercept


class DecisionEngine:
    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        dcfg = cfg.curiosity_of("decision")
        self.ask_threshold = float(dcfg.get("ask_threshold", 6.0))
        self.weights: dict[str, float] = dcfg.get("weights", {})

    def decide(
        self,
        obj: Optional[ObjectPercept],
        human: dict[str, Any],
        task: dict[str, Any],
        exploration_cost: float = 0.5,
    ) -> Decision:
        if obj is None:
            return Decision(path=DecisionPath.EXPLORE, score=0.0,
                            detail={"reason": "no_object"})

        danger = obj.danger_level
        dynamicity = 1.0 if obj.state != "static" else 0.0

        human_present = human.get("present", False)
        human_idle = human.get("idle", False)
        if human_present and human_idle:
            s_human = 10.0
        elif human_present:
            s_human = 5.0
        else:
            s_human = 2.0

        urgency = 1.0 if task.get("urgent", False) else 0.0

        scores = {
            "risk_level": float(np_clip(danger * 10.0, 0, 10)),
            "human_idle": s_human,
            "exploration_cost": float(np_clip(exploration_cost * 10.0, 0, 10)),
            "task_priority": float(np_clip((1.0 - urgency) * 10.0, 0, 10)),
            "object_dynamicity": float(np_clip((1.0 - dynamicity) * 10.0, 0, 10)),
        }
        total = sum(self.weights.get(k, 0.0) * v for k, v in scores.items())
        path = DecisionPath.ASK_HUMAN if total >= self.ask_threshold else DecisionPath.EXPLORE
        return Decision(path=path, score=float(total), detail=scores)


def np_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
