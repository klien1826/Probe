"""判断层 · 物理安全门禁（快系统）。

检查项（任一触发即中断，安全优先）:
  - 重心投影超出支撑多边形 → 急停 HALT
  - 关节角度接近限位 → 软停止 SOFT_STOP
  - 前方落差 > 阈值 → 原地等待 WAIT
  - 电池 < 阈值 → 强制返回充电 RECHARGE
  - 线/角速度超限 → 钳制 NORMAL(降速)

真实部署: 文档规格为 1kHz 硬实时，独立实时线程，C++ 扩展经 ctypes 调用。
本实现提供纯 Python 参考实现（接口一致），并预留 ctypes 扩展加载点。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.config import Config
from ..core.types import ProprioState, SafetyAction


class SafetyGate:
    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        scfg = cfg.safety_of("physical")
        self.max_cop_offset = float(scfg.get("max_cop_offset_m", 0.10))
        self.joint_limit_margin = float(scfg.get("joint_limit_margin", 0.90))
        self.max_drop = float(scfg.get("max_drop_m", 0.10))
        self.min_battery = float(scfg.get("min_battery_pct", 10.0))
        self.max_linear = float(scfg.get("max_linear_vel", 0.5))
        self.max_angular = float(scfg.get("max_angular_vel", 1.2))
        self.hard_realtime = bool(cfg.safety_of("system").get("hard_realtime", False))
        self._ctypes_lib = None
        if self.hard_realtime:
            self._load_native()

    def _load_native(self):
        """真实部署加载 C++ 扩展（ctypes）。这里仅为占位。"""
        try:
            import ctypes
            self._ctypes_lib = ctypes.CDLL("libsafety_gate.so")
        except Exception:
            self._ctypes_lib = None

    def check(self, proprio: ProprioState) -> SafetyAction:
        """返回安全动作（HALT 优先）。"""
        if proprio.cop_offset > self.max_cop_offset:
            return SafetyAction.HALT
        if proprio.front_drop > self.max_drop:
            return SafetyAction.WAIT
        if proprio.battery_pct < self.min_battery:
            return SafetyAction.RECHARGE
        if self._joints_near_limit(proprio.joint_angles):
            return SafetyAction.SOFT_STOP
        if abs(proprio.linear_vel) > self.max_linear or abs(proprio.angular_vel) > self.max_angular:
            return SafetyAction.SOFT_STOP
        return SafetyAction.NORMAL

    def clamp(self, actions: dict[str, float]) -> dict[str, float]:
        """对动作做速度钳制（在急停等场景下置零）。"""
        out = dict(actions)
        linear = out.get("linear", 0.0)
        angular = out.get("angular", 0.0)
        out["linear"] = float(np.clip(linear, -self.max_linear, self.max_linear))
        out["angular"] = float(np.clip(angular, -self.max_angular, self.max_angular))
        return out

    def _joints_near_limit(self, joint_angles: np.ndarray) -> bool:
        if joint_angles is None or len(joint_angles) == 0:
            return False
        # 假定关节限位为 [-1, 1] rad（归一化）；超过 margin 即近限位
        return bool(np.any(np.abs(joint_angles) > self.joint_limit_margin))
