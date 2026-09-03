"""Brainbot 适配器 · 载荷转换。

把 Brainbot 的 ObservationMessage.payload（dict）转成认知大脑的 Observation。
支持两种来源:
  1. 本工程仿真直接输出的结构化载荷（proprio/scene/human/task）
  2. 上游 Brainbot/GR00T 风格载荷（state.* / observation.images.* / language_instruction）
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.types import ObjectPercept, Observation, ProprioState, ScenePercept


def payload_to_observation(payload: dict[str, Any]) -> Observation:
    """把 Brainbot 观测载荷映射为认知大脑的 Observation。"""
    if "proprio" in payload or "scene" in payload:
        return _structured(payload)

    # ---- GR00T 风格载荷 ----
    proprio = ProprioState(
        joint_angles=_as_vec(payload.get("state.joints", [])),
        torques=_as_vec(payload.get("state.torques", [])),
        collision=bool(payload.get("state.collision", False)),
        battery_pct=float(payload.get("state.battery", 100.0)),
    )
    task_desc = payload.get("annotation.human.task_description", "")
    if isinstance(task_desc, (list, tuple)):
        task_desc = task_desc[0] if task_desc else ""
    scene = None
    return Observation(
        proprio=proprio,
        scene=scene,
        human={"present": False, "idle": False, "answer": None},
        task={"urgent": False, "type": str(task_desc or "generic")},
    )


def _structured(payload: dict[str, Any]) -> Observation:
    p = payload.get("proprio", {})
    proprio = ProprioState(
        joint_angles=_as_vec(p.get("joint_angles", [])),
        torques=_as_vec(p.get("torques", [])),
        collision=bool(p.get("collision", False)),
        battery_pct=float(p.get("battery_pct", 100.0)),
        pose=tuple(p.get("pose", (0.0, 0.0, 0.0))),
        linear_vel=float(p.get("linear_vel", 0.0)),
        angular_vel=float(p.get("angular_vel", 0.0)),
        cop_offset=float(p.get("cop_offset", 0.0)),
        front_drop=float(p.get("front_drop", 0.0)),
    )
    s = payload.get("scene") or {}
    objects = []
    for od in s.get("objects", []):
        objects.append(ObjectPercept(
            object_id=od.get("object_id", "obj"),
            category=od.get("category"),
            color=tuple(od.get("color", (0.5, 0.5, 0.5))),
            size=float(od.get("size", 0.5)),
            state=od.get("state", "static"),
            touchable=bool(od.get("touchable", True)),
            danger_level=float(od.get("danger_level", 0.0)),
            feature=_as_vec(od.get("feature", []), 16),
            position=tuple(od.get("position", (0.0, 0.0))),
        ))
    scene = ScenePercept(
        scene_vector=_as_vec(s.get("scene_vector", []), 256),
        objects=objects,
        timestamp=float(s.get("timestamp", 0.0)),
    )
    human = payload.get("human", {})
    task = payload.get("task", {})
    return Observation(
        proprio=proprio,
        scene=scene,
        human={"present": bool(human.get("present", False)),
               "idle": bool(human.get("idle", False)),
               "answer": human.get("answer")},
        task={"urgent": bool(task.get("urgent", False)),
              "type": task.get("type", "generic")},
        audio=payload.get("audio"),
    )


def _as_vec(v: Any, size: int | None = None) -> np.ndarray:
    arr = np.asarray(v if v is not None else [], dtype=np.float32).reshape(-1)
    if size is not None:
        if arr.shape[0] < size:
            arr = np.concatenate([arr, np.zeros(size - arr.shape[0], dtype=np.float32)])
        else:
            arr = arr[:size]
    return arr


def brain_action_to_payload(brain_action) -> dict[str, Any]:
    """把大脑动作转成 Brainbot 动作载荷（dict[str, float]）。"""
    return dict(brain_action.actions)
