"""仿真 · MuJoCo 物理后端桥接。

把 MuJoCo 物理仿真接入认知闭环，作为 URLab+Schola 之外的 CPU 可运行物理后端。
  - 差速底盘运动学: v_l = v - ω·w/2, v_r = v + ω·w/2
  - 提供本体感知（位姿/速度/碰撞）与场景感知（物体位姿跟踪）
  - 真实部署可替换为 URLab/Schola 的 Gymnasium 环境，接口保持不变
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.core.types import ObjectPercept, Observation, ProprioState, ScenePercept

MJCF_PATH = Path(__file__).resolve().parent / "mujoco_models" / "brainbot_diffdrive.xml"
WHEEL_BASE = 0.32          # 左右轮距（米）
WHEEL_RADIUS = 0.055
SENSOR_RANGE = 3.0


class MuJoCoBridge:
    def __init__(self, mjcf_path: str | Path = MJCF_PATH, seed: int = 0):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        # 机器人本体/目标物体映射（物体以额外 body 加入仿真）
        self._objects: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def reset(self, robot_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.data.qpos[:] = self.model.qpos0
        # freejoint 的 qpos: [x, y, z, qw, qx, qy, qz]
        self.data.qpos[0] = robot_pose[0]
        self.data.qpos[1] = robot_pose[1]
        self.data.qpos[3] = np.cos(robot_pose[2] / 2)
        self.data.qpos[6] = np.sin(robot_pose[2] / 2)
        self.mj.mj_forward(self.model, self.data)

    def add_object(self, object_id: str, pos: tuple[float, float],
                   category: str | None = None, color=(0.9, 0.2, 0.2),
                   size: float = 0.2, state: str = "static",
                   danger: float = 0.1, touchable: bool = True,
                   label_visible: bool = False):
        """向仿真世界添加一个物体（简单 geom body）。"""
        spec = {
            "id": object_id, "category": category, "color": tuple(color),
            "size": size, "state": state, "danger": danger,
            "touchable": touchable, "pos": tuple(pos),
            "label_visible": label_visible, "revealed": category is None,
        }
        self._objects.append(spec)

    # ------------------------------------------------------------------
    def step(self, linear: float, angular: float, dt: float = 0.05):
        """差速运动学 → 轮速伺服（Python P 控制器）→ 物理步进。"""
        v_des_l = (linear - angular * WHEEL_BASE / 2) / WHEEL_RADIUS
        v_des_r = (linear + angular * WHEEL_BASE / 2) / WHEEL_RADIUS
        # 当前轮速（qvel 前 6 维是 freejoint，之后是各关节）
        v_l = self.data.qvel[6]
        v_r = self.data.qvel[7]
        kp = 8.0
        self.data.ctrl[0] = float(np.clip(kp * (v_des_l - v_l), -5.0, 5.0))
        self.data.ctrl[1] = float(np.clip(kp * (v_des_r - v_r), -5.0, 5.0))
        self.mj.mj_step(self.model, self.data, nstep=max(1, int(dt / self.model.opt.timestep)))

    def get_observation(self, human: dict | None = None,
                        battery_pct: float = 100.0) -> Observation:
        """构建认知大脑可消费的 Observation（MuJoCo 物理后端）。"""
        x, y = self.data.qpos[0], self.data.qpos[1]
        th = self._heading()
        # 碰撞检测
        collision = any(self.data.contact[i].dist < 0 for i in range(self.data.ncon))
        proprio = ProprioState(
            joint_angles=np.asarray(self.data.qpos[7:13], dtype=np.float32),
            torques=np.zeros(6, dtype=np.float32),
            collision=collision,
            battery_pct=battery_pct,
            pose=(float(x), float(y), float(th)),
            linear_vel=float(np.linalg.norm(self.data.qvel[:2])),
            angular_vel=float(self.data.qvel[5]),
            cop_offset=0.0,
            front_drop=0.0,
        )
        visible: list[ObjectPercept] = []
        for o in self._objects:
            ox, oy = o["pos"]
            dist = np.hypot(ox - x, oy - y)
            if dist > SENSOR_RANGE:
                continue
            visible.append(ObjectPercept(
                object_id=o["id"],
                category=o["category"] if o["revealed"] else None,
                color=o["color"], size=o["size"], state=o["state"],
                touchable=o["touchable"], danger_level=o["danger"],
                feature=np.asarray([*o["color"], o["size"],
                                    1.0 if o["state"] == "moving" else 0.0,
                                    o["danger"], 1.0 if o["label_visible"] else 0.0]
                                   + [0.0] * 10, dtype=np.float32),
                position=(float(ox - x), float(oy - y)),
            ))
        scene_vec = np.zeros(256, dtype=np.float32)
        if visible:
            feats = np.concatenate([v.feature for v in visible])
            scene_vec[: min(len(feats), 192)] = feats[: min(len(feats), 192)]
        scene_vec[192] = battery_pct / 100.0
        scene_vec[193] = min(1.0, len(visible) / 5.0)
        n = np.linalg.norm(scene_vec)
        scene = ScenePercept(scene_vector=scene_vec / n if n > 1e-9 else scene_vec,
                             objects=visible, timestamp=0.0)
        return Observation(
            proprio=proprio, scene=scene,
            human=human or {"present": True, "idle": True, "answer": None},
            task={"urgent": False, "type": "active_learning"},
        )

    def reveal(self, object_id: str):
        for o in self._objects:
            if o["id"] == object_id:
                o["revealed"] = True

    def _heading(self) -> float:
        q = self.data.qpos[3:7]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                1 - 2 * (q[2] ** 2 + q[3] ** 2)))
