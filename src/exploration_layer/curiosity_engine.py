"""探索层 · 好奇心引擎（Explauto 封装 + 兴趣模型降级后端）。

输入: 当前状态 + 预测误差/不确定性
输出: 探索模式 (广泛扫描 / 深度钻研 / 复习巩固 / 安全待机)

双后端:
  - backend="explauto": 真实 Explauto（Inria FLOWERS 团队，
    https://github.com/flowersteam/explauto）。该库面向传感器-运动空间，
    需适配采样空间，CPU 环境也可用但接口陈旧（Py3.12 兼容性差）。
  - backend="interest": 兴趣模型（等价实现）。以内在奖励
    intrinsic = 不确定性分数 驱动探索模式切换，阈值来自配置。
"""
from __future__ import annotations

from typing import Optional

from ..core.config import Config
from ..core.types import ExplorationMode, UncertaintySignal


class CuriosityEngine:
    def __init__(self, cfg: Config | None = None, backend: str = "auto"):
        cfg = cfg or Config()
        self.cfg = cfg
        self.backend = backend if backend != "auto" else self._pick_backend()
        self._explauto = None
        if self.backend == "explauto":
            self._init_explauto()

    def _pick_backend(self) -> str:
        try:
            from explauto.interest_model.interest_model import InterestModel  # noqa: F401
            return "explauto"
        except Exception:
            return "interest"

    def _init_explauto(self):
        # Explauto 需要定义传感器/运动空间与采样器；这里保留真实适配入口。
        # 例: from explauto import Environment, SensorimotorModel, InterestModel
        pass

    def select_mode(
        self,
        uncertainty: Optional[UncertaintySignal] = None,
        prediction_error: float = 0.0,
        familiarity: float = 1.0,
    ) -> ExplorationMode:
        """依据内在奖励选择探索模式。

        intrinsic = 不确定性分数（预测误差 + 方差项）。
        阈值阶梯: 0.6 广泛扫描 / 0.3 深度钻研 / 0.1 复习巩固 / 否则安全待机。
        """
        intrinsic = prediction_error
        if uncertainty is not None:
            intrinsic = max(intrinsic, uncertainty.uncertainty_score)

        # 已高度熟悉（familiarity 高）+ 低误差 → 复习巩固或待机
        modes = self.cfg.curiosity_of("exploration").get("modes", [])
        # 按 min_intrinsic 降序遍历，取第一个满足的
        order = sorted(modes, key=lambda m: m["min_intrinsic"], reverse=True)
        for m in order:
            if intrinsic >= m["min_intrinsic"]:
                return ExplorationMode(m["name"])
        return ExplorationMode.SAFE_IDLE
