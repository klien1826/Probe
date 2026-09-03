"""Brainbot 适配器 · 命令提供者。

把七层认知大脑实现为 Brainbot 的 CommandProvider：
  - 上游接口: CommandProvider.compute_command(observation: ObservationMessage) -> ActionMessage
  - 本实现直接内嵌大脑（单进程），也可改为通过 ZMQ 连接 policy_server
"""
from __future__ import annotations

from typing import Any, Mapping

from brainbot_core.proto import ActionMessage, ObservationMessage

from ..core.brain import CognitiveBrain
from .payload import payload_to_observation


class CuriosityCommandProvider:
    """把认知大脑作为命令提供者，供 Brainbot 命令服务调度。"""

    def __init__(self, brain: CognitiveBrain | None = None):
        self.brain = brain or CognitiveBrain()
        self._instruction: str | None = None

    def prepare(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def wants_full_observation(self) -> bool:
        return True

    def set_instruction(self, instruction: str) -> None:
        self._instruction = instruction

    def clear_instruction(self) -> None:
        self._instruction = None

    def compute_command(self, observation: ObservationMessage) -> ActionMessage:
        """上游 Brainbot 回调：观测 → 动作。"""
        payload = dict(observation.payload)
        obs = payload_to_observation(payload)
        if self._instruction:
            obs.task["type"] = self._instruction
        brain_action = self.brain.think(obs)
        return ActionMessage(actions=dict(brain_action.actions))
