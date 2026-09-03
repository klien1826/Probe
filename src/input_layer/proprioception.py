"""输入层 · 本体感知（本体感觉）。

输入: 关节编码器 + IMU + 里程计 + 电量（来自仿真或真实硬件）
输出: ProprioState
频率: 100 Hz
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.config import Config
from ..core.types import ProprioState


class Proprioception:
    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        scfg = cfg.safety_of("physical")
        self.min_battery = scfg.get("min_battery_pct", 10.0)

    def read(self, raw: dict[str, Any]) -> ProprioState:
        """raw 来自硬件驱动/仿真后端。字段缺失时提供安全默认。"""
        return ProprioState(
            joint_angles=np.asarray(raw.get("joint_angles", np.zeros(6)), dtype=np.float32),
            torques=np.asarray(raw.get("torques", np.zeros(6)), dtype=np.float32),
            collision=bool(raw.get("collision", False)),
            battery_pct=float(raw.get("battery_pct", 100.0)),
            pose=tuple(raw.get("pose", (0.0, 0.0, 0.0))),
            linear_vel=float(raw.get("linear_vel", 0.0)),
            angular_vel=float(raw.get("angular_vel", 0.0)),
            cop_offset=float(raw.get("cop_offset", 0.0)),
            front_drop=float(raw.get("front_drop", 0.0)),
        )
