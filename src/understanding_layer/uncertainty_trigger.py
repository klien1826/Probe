"""理解层 · 预测方差触发器（替代"自觉知"）。

触发条件（任一满足）:
  1. prediction_error > 好奇阈值 (默认 0.30)
  2. 物体 ID 在记忆库中无匹配（对象新颖性）

输出: {"trigger": bool, "object_id": str|None, "uncertainty_score": float}
"""
from __future__ import annotations

from typing import Optional

from ..core.config import Config
from ..core.types import PredictionResult, UncertaintySignal


class UncertaintyTrigger:
    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        pcfg = cfg.curiosity_of("prediction")
        self.error_threshold = float(pcfg.get("curiosity_error_threshold", 0.30))
        ucfg = cfg.curiosity_of("uncertainty")
        self.trigger_on_unknown = bool(ucfg.get("trigger_on_unknown_object", True))

    def evaluate(
        self,
        pred: PredictionResult,
        object_known: bool,
        object_id: Optional[str] = None,
    ) -> UncertaintySignal:
        reasons: list[str] = []
        if pred.prediction_error > self.error_threshold:
            reasons.append(
                f"prediction_error={pred.prediction_error:.3f}>{self.error_threshold}")
        if self.trigger_on_unknown and not object_known:
            reasons.append("object_unknown_in_memory")

        triggered = len(reasons) > 0
        return UncertaintySignal(
            trigger=triggered,
            object_id=object_id,
            uncertainty_score=float(pred.prediction_error),
            reason="; ".join(reasons),
        )
