"""共享类型定义：七层认知架构各层之间的数据契约。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class ExplorationMode(str, Enum):
    BROAD_SCAN = "broad_scan"
    DEEP_DIVE = "deep_dive"
    REVIEW = "review_consolidate"
    SAFE_IDLE = "safe_idle"


class DecisionPath(str, Enum):
    ASK_HUMAN = "ask_human"
    EXPLORE = "explore"


class SafetyAction(str, Enum):
    NORMAL = "normal"
    SOFT_STOP = "soft_stop"
    HALT = "halt"
    WAIT = "wait"
    RECHARGE = "recharge"


@dataclass
class ObjectPercept:
    """一个被感知的物体对象。"""
    object_id: str
    category: Optional[str]          # None = 未知类别
    color: tuple[float, float, float]
    size: float
    state: str = "static"            # static | moving | openable ...
    touchable: bool = True
    danger_level: float = 0.0        # 0~1
    feature: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))
    position: tuple[float, float] = (0.0, 0.0)


@dataclass
class ScenePercept:
    """输入层 → 理解层的场景感知。"""
    scene_vector: np.ndarray          # 256 维场景特征
    objects: list[ObjectPercept]
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    timestamp: float = 0.0


@dataclass
class ProprioState:
    joint_angles: np.ndarray
    torques: np.ndarray
    collision: bool
    battery_pct: float
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)   # x, y, theta
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    cop_offset: float = 0.0          # 重心投影偏移（米）
    front_drop: float = 0.0          # 前方落差（米）


@dataclass
class PredictionResult:
    """世界模型预测输出。"""
    z_pred: np.ndarray
    z_real: np.ndarray
    mse: float
    variance: float
    prediction_error: float          # 0~1 归一化
    attended_object_id: Optional[str] = None


@dataclass
class UncertaintySignal:
    trigger: bool
    object_id: Optional[str]
    uncertainty_score: float
    reason: str = ""


@dataclass
class ActionPlan:
    action: str                      # "move" | "rotate" | "approach" | "inspect" | "ask" | "wait"
    params: dict[str, Any] = field(default_factory=dict)
    expected_curiosity_gain: float = 0.0
    risk: float = 0.0
    cost: float = 0.0


@dataclass
class Decision:
    path: DecisionPath
    score: float
    detail: dict[str, float] = field(default_factory=dict)
    plan: Optional[ActionPlan] = None
    question: Optional[str] = None


@dataclass
class BrainAction:
    """大脑最终输出（兼容 Brainbot ActionMessage.actions 语义）。"""
    actions: dict[str, float] = field(default_factory=dict)
    text: Optional[str] = None
    decision: Optional[Decision] = None
    memory_write: bool = False
    target_id: Optional[str] = None          # 本次动作关注的目标物体 ID


@dataclass
class Observation:
    """大脑闭环的输入。"""
    proprio: ProprioState
    scene: Optional[ScenePercept] = None
    human: dict[str, Any] = field(default_factory=dict)   # {"present": bool, "idle": bool, "answer": str|None}
    task: dict[str, Any] = field(default_factory=dict)    # {"urgent": bool, "type": str}
    audio: Optional[dict[str, Any]] = None
