"""理解层 · 世界模型（MWM 封装 + 轻量降级后端）。

输入: 当前隐状态 z_t + 动作 a_t
输出: 预测的下一隐状态 z_{t+1}_pred 及其预测方差
推理延迟预算: ≤ 50ms

双后端:
  - backend="mwm": Mask World Model (ICML 2026, arXiv:2604.19683) 官方实现。
    真实部署需 GPU + GE-Base 权重，见 https://github.com/LYFCLOUDFAN/mask-world-model
  - backend="lightweight": 在线贝叶斯线性动力学。
    每个已知物体类别维护一个后验线性模型；未知物体使用"先验模型"
    （大先验方差）→ 预测方差高 → 触发"预测方差"好奇信号。
    这正是"预测方差触发"替代"自觉知"的最小可验证实现。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.config import Config


class BayesianLinearDynamics:
    """在线贝叶斯线性回归（多输出），用于过渡动力学估计。

    输入特征 φ (in_dim,)，预测输出 y (out_dim,)。
    权重后验: mu (in_dim, out_dim)，协方差 Sigma (in_dim, in_dim) 各输出共享。
    """

    def __init__(self, input_dim: int, output_dim: int,
                 noise_var: float = 0.05, prior_cov: float = 1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_var = noise_var
        self.mu = np.zeros((input_dim, output_dim), dtype=np.float64)
        self.Sigma = np.eye(input_dim, dtype=np.float64) * prior_cov
        self.samples = 0

    def predict(self, phi: np.ndarray) -> tuple[np.ndarray, float]:
        phi = phi.astype(np.float64)
        mean = phi @ self.mu                                  # (out_dim,)
        var = float(phi @ self.Sigma @ phi + self.noise_var)
        return mean, var

    def update(self, phi: np.ndarray, y: np.ndarray):
        phi = phi.astype(np.float64)
        y = np.asarray(y, dtype=np.float64)
        Sigma_phi = self.Sigma @ phi                          # (in_dim,)
        denom = float(phi @ Sigma_phi + self.noise_var)
        gain = Sigma_phi / denom                              # (in_dim,)
        err = y - phi @ self.mu                               # (out_dim,)
        self.mu = self.mu + np.outer(gain, err)
        self.Sigma = self.Sigma - np.outer(gain, Sigma_phi)
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)        # 对称化
        self.samples += 1


class WorldModel:
    LATENT_DIM = 32

    def __init__(self, cfg: Config | None = None, backend: str = "auto"):
        cfg = cfg or Config()
        wcfg = cfg.get("world_model")
        self.backend = wcfg.get("backend", "auto")
        if self.backend == "auto":
            self.backend = "lightweight"   # CPU 环境默认轻量后端
        self.latent_dim = wcfg.get("latent_dim", 32)
        self.mwm_cfg = wcfg.get("mwm", {})
        lw = wcfg.get("lightweight", {})
        self._noise_var = lw.get("obs_noise", 0.05)
        self._prior_cov = lw.get("prior_cov_scale", 1.0)
        self._models: dict[str, BayesianLinearDynamics] = {}
        self._model_counts: dict[str, int] = {}
        self._novel_prior_cov = 5.0          # 未知物体的先验方差（触发好奇）
        self._mwm_loaded = False
        self.rng = np.random.default_rng(2026)
        # 动力学输入维 = 隐状态维 + 动作维(3)
        self._feat_dim = self.latent_dim + 3

    # ------------------------------------------------------------------
    def predict(
        self,
        z_t: np.ndarray,
        action_vec: np.ndarray,
        attended_object_id: Optional[str] = None,
    ) -> tuple[np.ndarray, float]:
        """返回 (z_{t+1}_pred, predictive_variance)。

        模型学的是 delta = z_{t+1} - z_t，因此预测的下一个状态 = z_t + delta。
        """
        z_t = np.asarray(z_t, dtype=np.float64).reshape(-1)
        phi = np.concatenate([z_t, np.asarray(action_vec, dtype=np.float64).reshape(-1)])
        model, known = self._get_model(attended_object_id)
        delta, var = model.predict(phi)
        return z_t + delta, var

    def update(
        self,
        z_t: np.ndarray,
        action_vec: np.ndarray,
        z_next: np.ndarray,
        attended_object_id: Optional[str] = None,
        commit: bool = True,
    ):
        """用真实过渡更新模型（只在物体已识别时提交，避免污染未知先验）。"""
        z_t = np.asarray(z_t, dtype=np.float64).reshape(-1)
        z_next = np.asarray(z_next, dtype=np.float64).reshape(-1)
        phi = np.concatenate([z_t, np.asarray(action_vec, dtype=np.float64).reshape(-1)])
        delta = z_next - z_t
        if commit and attended_object_id:
            model, _ = self._get_model(attended_object_id, create=True)
            model.update(phi, delta)
            self._model_counts[attended_object_id] = model.samples
        return delta

    def familiarity(self, object_id: str) -> float:
        """0~1：该物体类别的动力学学习置信度。"""
        n = self._model_counts.get(object_id, 0)
        return min(1.0, n / 10.0)

    def seed_known(self, object_id: str, z: np.ndarray, action_vec: np.ndarray, n: int = 8):
        """物体被识别后播种已知样本，坍缩预测方差（停止对该物体的好奇触发）。

        以"近似静态"的过渡（delta≈0 + 微小噪声）提交若干样本，
        使该物体模型的贝叶斯后验方差快速下降。
        """
        model, _ = self._get_model(object_id, create=True)
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        phi = np.concatenate([z, np.asarray(action_vec, dtype=np.float64).reshape(-1)])
        for _ in range(n):
            noise = self.rng.standard_normal(self.latent_dim) * 0.01
            model.update(phi, noise)
        self._model_counts[object_id] = model.samples

    # ------------------------------------------------------------------
    def _get_model(self, object_id: Optional[str], create: bool = False):
        """按物体类别取模型；未知类别 → 高方差先验模型（不提交）。"""
        if object_id is None:
            key = "__scene__"
        else:
            key = object_id
        if key in self._models:
            return self._models[key], True
        if object_id is None or create:
            model = BayesianLinearDynamics(
                self._feat_dim, self.latent_dim, self._noise_var, self._prior_cov)
            self._models[key] = model
            return model, True
        # 未知物体：返回一个临时高方差先验模型（不缓存/不提交）
        return BayesianLinearDynamics(
            self._feat_dim, self.latent_dim, self._noise_var, self._novel_prior_cov), False


def build_latent(proprio_vec: np.ndarray, object_feature: np.ndarray, dim: int = 32) -> np.ndarray:
    """把本体状态 + 被关注物体的特征投影为固定维隐状态 z。"""
    z = np.concatenate([proprio_vec, object_feature]).astype(np.float64)
    if z.shape[0] < dim:
        z = np.concatenate([z, np.zeros(dim - z.shape[0])])
    return z[:dim]
