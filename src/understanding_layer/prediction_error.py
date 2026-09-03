"""理解层 · 预测误差计算。

prediction_error = ||z_pred - z_real||²   （MSE）
碰撞加权: collision_flag == True → error *= 2.0
叠加世界模型预测方差项（预测方差触发），归一化到 0~1。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.config import Config
from ..core.types import PredictionResult


class PredictionError:
    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        pcfg = cfg.curiosity_of("prediction")
        self.normalizer = float(pcfg.get("error_normalizer", 1.0))
        self.collision_weight = float(pcfg.get("collision_error_weight", 2.0))
        ucfg = cfg.curiosity_of("uncertainty")
        self.variance_weight = float(ucfg.get("variance_weight", 0.5))

    def compute(
        self,
        z_pred: np.ndarray,
        z_real: np.ndarray,
        variance: float,
        collision: bool = False,
        attended_object_id: Optional[str] = None,
    ) -> PredictionResult:
        z_pred = np.asarray(z_pred, dtype=np.float64)
        z_real = np.asarray(z_real, dtype=np.float64)
        mse = float(np.mean((z_pred - z_real) ** 2))

        # 方差项归一化（相对隐状态量纲）
        var_norm = float(np.tanh(variance / 10.0))

        error = mse / self.normalizer + self.variance_weight * var_norm
        if collision:
            error *= self.collision_weight
        error = float(np.clip(error, 0.0, 1.0))

        return PredictionResult(
            z_pred=z_pred.astype(np.float32),
            z_real=z_real.astype(np.float32),
            mse=mse,
            variance=variance,
            prediction_error=error,
            attended_object_id=attended_object_id,
        )
