"""仿真环境 · 轻量 2D 世界（七层认知闭环的驱动载体）。

在 Unreal/URLab 不可用的 CPU 沙箱中，本仿真提供：
  - 移动机器人（位姿/电量/速度）
  - 带"真值"的物体（类别可能未知、可能带可见标签、可能危险/动态）
  - 人类（在场/空闲状态，可回答问题）
  - 感知模型：可见物体 → ScenePercept（类别未知时 category=None）
  - 物理：简单运动学 + 碰撞

揭示规则（决定"未知→已知"的两种途径，对应双路径决策）:
  - 探索路径: 机器人对带可见标签的物体执行 inspect（且足够近）→ 读标签 → 类别揭示
  - 提问路径: 机器人执行 ask 且人类在场 → 人类给出真值回答 → 类别揭示

真实部署时，本仿真的输入/输出可替换为 URLab+Schola 的 Gymnasium 环境
（src/simulation/mujoco_bridge.py 提供 MuJoCo 物理后端入口）。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.core.types import BrainAction, ObjectPercept, Observation, ProprioState, ScenePercept

SENSOR_RANGE = 2.0
APPROACH_RANGE = 0.4
INSPECT_RANGE = 0.5
COLLISION_RANGE = 0.12
FEATURE_DIM = 16


class WorldSim:
    def __init__(self, seed: int = 20260903, human_present: bool = True, human_idle: bool = True,
                 battery_pct: float = 100.0, objects: Optional[list[dict[str, Any]]] = None):
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self._init_battery = battery_pct
        self.robot = {
            "pose": (0.0, 0.0, 0.0),
            "battery": battery_pct,
            "linear": 0.0,
            "angular": 0.0,
            "heading": 0.0,
        }
        self.human = {"present": human_present, "idle": human_idle}
        # 物体真值表：category 为真值；revealed=False 表示机器人尚未识别
        default_objects = [
            {"id": "obj_red_box", "category": "box", "color": (0.9, 0.1, 0.1), "size": 0.5,
             "state": "static", "danger": 0.1, "touchable": True, "pos": (1.5, 0.3),
             "label_visible": False},
            {"id": "obj_blue_bottle", "category": "bottle", "color": (0.1, 0.3, 0.9), "size": 0.3,
             "state": "static", "danger": 0.2, "touchable": True, "pos": (1.2, -0.6),
             "label_visible": True},
            {"id": "obj_green_ball", "category": "ball", "color": (0.2, 0.8, 0.2), "size": 0.4,
             "state": "moving", "danger": 0.05, "touchable": True, "pos": (2.0, 0.8),
             "label_visible": False},
        ]
        self.objects = []
        for spec in (objects or default_objects):
            self.objects.append({
                "id": spec["id"],
                "category": spec["category"],
                "color": tuple(spec["color"]),
                "size": float(spec["size"]),
                "state": spec.get("state", "static"),
                "danger": float(spec.get("danger", 0.0)),
                "touchable": bool(spec.get("touchable", True)),
                "pos": (float(spec["pos"][0]), float(spec["pos"][1])),
                "label_visible": bool(spec.get("label_visible", False)),
                "revealed": bool(spec.get("revealed", False)),
            })
        self._pending_answer: Optional[str] = None

    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        self.time = 0.0
        self.robot["pose"] = (0.0, 0.0, 0.0)
        self.robot["battery"] = self._init_battery
        self.robot["linear"] = 0.0
        self.robot["angular"] = 0.0
        for o in self.objects:
            o["revealed"] = False
        self._pending_answer = None
        return self._observe()

    def step(self, brain_action: BrainAction) -> Observation:
        self.time += 1.0 / 10.0          # 10 Hz 仿真步
        self.robot["battery"] = max(0.0, self.robot["battery"] - 0.02)

        code = brain_action.actions.get("action", 5.0)
        target = self._get_object(brain_action.target_id) or self._nearest_unknown()

        # ---- 执行动作 ----
        if code == 0.0:      # move: 漫游
            self._move_forward(0.12)
        elif code == 1.0:    # rotate
            self._rotate(0.4)
        elif code == 2.0:    # approach
            if target is not None:
                self._approach(target)
        elif code == 3.0:    # inspect
            self._inspect(target)
        elif code == 4.0:    # ask
            self._ask(target)

        # ---- 动态物体轻微移动 ----
        for o in self.objects:
            if o["state"] == "moving":
                o["pos"] = (o["pos"][0] + 0.005 * self.rng.standard_normal(),
                            o["pos"][1] + 0.005 * self.rng.standard_normal())

        return self._observe()

    # ------------------------------------------------------------------
    # 机器人运动
    # ------------------------------------------------------------------
    def _move_forward(self, step: float):
        x, y, th = self.robot["pose"]
        x += step * np.cos(th)
        y += step * np.sin(th)
        self.robot["pose"] = (x, y, th)
        self.robot["linear"] = step

    def _rotate(self, dtheta: float):
        x, y, th = self.robot["pose"]
        self.robot["pose"] = (x, y, th + dtheta)
        self.robot["angular"] = dtheta

    def _approach(self, target: dict):
        x, y, th = self.robot["pose"]
        tx, ty = target["pos"]
        dx, dy = tx - x, ty - y
        dist = np.hypot(dx, dy)
        if dist < APPROACH_RANGE:
            self.robot["linear"] = 0.0
            return
        desired = np.arctan2(dy, dx)
        # 转向 + 前进
        dtheta = np.arctan2(np.sin(desired - th), np.cos(desired - th))
        self.robot["pose"] = (x, y, th + 0.5 * dtheta)
        step = 0.12 if abs(dtheta) < 0.5 else 0.02
        nx = x + step * np.cos(self.robot["pose"][2])
        ny = y + step * np.sin(self.robot["pose"][2])
        self.robot["pose"] = (nx, ny, self.robot["pose"][2])
        self.robot["linear"] = step

    def _inspect(self, target: Optional[dict]):
        if target is None:
            return
        dist = self._dist_to(target)
        if dist <= INSPECT_RANGE and target.get("label_visible"):
            target["revealed"] = True          # 读标签 → 类别揭示（探索路径）

    def _ask(self, target: Optional[dict]):
        if target is None:
            return
        if self.human["present"] and self.human["idle"]:
            self._pending_answer = _human_answer(target)
            target["revealed"] = True          # 人类回答 → 类别揭示（提问路径）

    # ------------------------------------------------------------------
    def _observe(self) -> Observation:
        x, y, th = self.robot["pose"]
        visible: list[ObjectPercept] = []
        for o in self.objects:
            ox, oy = o["pos"]
            dist = np.hypot(ox - x, oy - y)
            if dist > SENSOR_RANGE:
                continue
            # 类别仅在已揭示时可见
            category = o["category"] if o["revealed"] else None
            visible.append(ObjectPercept(
                object_id=o["id"],
                category=category,
                color=o["color"],
                size=o["size"],
                state=o["state"],
                touchable=o["touchable"],
                danger_level=o["danger"],
                feature=self._object_feature(o),
                position=(float(ox - x), float(oy - y)),
            ))

        # 碰撞检测（与最近物体过近）
        collision = any(self._dist_to(o) < COLLISION_RANGE for o in self.objects)

        scene = ScenePercept(
            scene_vector=self._scene_vector(visible),
            objects=visible,
            timestamp=self.time,
        )
        proprio = ProprioState(
            joint_angles=np.zeros(6),
            torques=np.zeros(6),
            collision=collision,
            battery_pct=self.robot["battery"],
            pose=(x, y, th),
            linear_vel=self.robot["linear"],
            angular_vel=self.robot["angular"],
            cop_offset=0.0,
            front_drop=0.0,
        )
        answer = self._pending_answer
        self._pending_answer = None
        return Observation(
            proprio=proprio,
            scene=scene,
            human={"present": self.human["present"], "idle": self.human["idle"],
                   "answer": answer},
            task={"urgent": False, "type": "active_learning"},
        )

    # ------------------------------------------------------------------
    def _object_feature(self, o: dict) -> np.ndarray:
        """确定性 16 维物体特征：颜色3 + 尺寸 + 状态 + 危险 + 标签可见性 + 填充。"""
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        f[:3] = o["color"]
        f[3] = o["size"]
        f[4] = 1.0 if o["state"] == "moving" else 0.0
        f[5] = o["danger"]
        f[6] = 1.0 if o.get("label_visible") else 0.0
        return f

    def _scene_vector(self, visible: list[ObjectPercept]) -> np.ndarray:
        """256 维场景向量：可见物体特征拼接 + 本体状态 + 固定投影。"""
        vec = np.zeros(256, dtype=np.float32)
        if visible:
            feats = np.concatenate([np.asarray(v.feature) for v in visible])
            vec[: min(len(feats), 192)] = feats[: min(len(feats), 192)]
        vec[192] = self.robot["battery"] / 100.0
        vec[193] = float(len(visible)) / 5.0
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-9 else vec

    def _dist_to(self, target: dict) -> float:
        x, y, _ = self.robot["pose"]
        return float(np.hypot(target["pos"][0] - x, target["pos"][1] - y))

    def _get_object(self, object_id: Optional[str]) -> Optional[dict]:
        if object_id is None:
            return None
        for o in self.objects:
            if o["id"] == object_id:
                return o
        return None

    def _nearest_unknown(self) -> Optional[dict]:
        unknown = [o for o in self.objects if not o["revealed"]]
        pool = unknown or self.objects
        if not pool:
            return None
        return min(pool, key=self._dist_to)

    def ground_truth(self) -> list[dict[str, Any]]:
        return [dict(o) for o in self.objects]


def _human_answer(target: dict) -> str:
    cat = target["category"] or "unknown"
    color_name = {0.9: "红", 0.1: "红", 0.8: "绿", 0.3: "蓝", 0.2: "绿"}.get(0.0, "")
    return f"这是一个{cat}"
