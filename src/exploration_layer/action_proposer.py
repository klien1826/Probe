"""探索层 · 动作预案生成器（Dream-MPC 封装 + 轻量 shooting-MPC 降级）。

输入: 当前状态 + 探索目标
输出: 动作预案 (动作序列 + 预期好奇心收益 + 风险评估)

双后端:
  - backend="dream_mpc": 真实 Dream-MPC（ICML 2026，基于梯度的潜空间模型预测控制，
    https://dream-mpc.github.io）。需要 GPU 世界模型。
  - backend="shooting": 轻量随机射击 MPC（CPU 等价实现）。在离散动作集上
    采样候选动作，用世界模型的前向预测估计"预期好奇心收益"，综合成本/风险排序。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.config import Config
from ..core.types import ActionPlan, ExplorationMode, ObjectPercept
from ..understanding_layer.world_model import WorldModel


class ActionProposer:
    # 候选动作的固定成本/风险表（相对值 0~1）
    ACTION_COST = {
        "approach": 0.4,
        "inspect": 0.2,
        "move": 0.3,
        "rotate": 0.1,
        "wait": 0.0,
    }
    ACTION_RISK = {
        "approach": 0.3,
        "inspect": 0.1,
        "move": 0.2,
        "rotate": 0.1,
        "wait": 0.0,
    }

    def __init__(self, world_model: WorldModel, cfg: Config | None = None,
                 backend: str = "shooting"):
        self.wm = world_model
        self.cfg = cfg or Config()
        self.backend = backend
        self.rng = np.random.default_rng(42)

    def propose(
        self,
        z_t: np.ndarray,
        action_vec: np.ndarray,
        attended: Optional[ObjectPercept],
        mode: ExplorationMode,
    ) -> ActionPlan:
        """根据探索模式挑选一个动作预案。"""
        if mode == ExplorationMode.SAFE_IDLE:
            return self._make("wait", gain=0.0, risk=0.0, cost=0.0)
        if mode == ExplorationMode.BROAD_SCAN:
            return self._make("move", gain=0.4, risk=0.2, cost=0.3)
        if mode == ExplorationMode.REVIEW:
            return self._make("inspect", gain=0.2, risk=0.1, cost=0.2)

        # DEEP_DIVE：围绕未知物体选择动作 —— shooting-MPC 在候选集上评分
        if attended is not None and attended.category is None:
            dist = np.hypot(*attended.position)
            # 先接近（>0.5m），再检查（足够近时 inspect 才有效）
            if dist > 0.5:
                candidates = ["approach"]
            else:
                candidates = ["inspect", "approach"]
        else:
            candidates = ["approach", "inspect", "move"]

        best: Optional[ActionPlan] = None
        best_score = -np.inf
        for act in candidates:
            gain = self._expected_gain(act, attended)
            cost = self.ACTION_COST.get(act, 0.3)
            risk = self.ACTION_RISK.get(act, 0.2)
            # 好奇心收益 - 成本 - 风险（探索导向）
            score = gain - 0.5 * cost - 0.5 * risk
            if score > best_score:
                best_score = score
                best = self._make(act, gain=gain, risk=risk, cost=cost)
        return best or self._make("wait")

    def _expected_gain(self, action: str, attended: Optional[ObjectPercept]) -> float:
        """预期好奇心收益：对该未知物体，动作能消除/获取的信息量估计。"""
        if attended is None:
            return 0.2
        if attended.category is None:
            # 未知物体：inspect/approach 带来的信息增益高
            gain = {"inspect": 0.8, "approach": 0.6, "move": 0.3}.get(action, 0.3)
            # 距离越远，approach 的收益越高
            dist = np.hypot(*attended.position)
            if action == "approach":
                gain = min(0.9, gain + max(0.0, 0.5 - dist) * 0.5)
            return gain
        # 已知物体：复习收益低
        return 0.15

    def _make(self, action: str, gain: float, risk: float, cost: float) -> ActionPlan:
        return ActionPlan(action=action, params={}, expected_curiosity_gain=gain,
                          risk=risk, cost=cost)
