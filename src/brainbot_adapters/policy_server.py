"""Brainbot 适配器 · AI 策略服务器。

将"四层大脑"封装为 Brainbot 可识别的 AI Provider：
  - 协议: ZMQ (REP/REQ) + msgpack（复用上游 brainbot_core.transport.BaseZMQServer）
  - 绑定: tcp://127.0.0.1:5555
  - 消息: 接收 {"endpoint":"get_action","data":{observation}} → 返回 {"action":{...}}

启动: python -m src.brainbot_adapters.policy_server
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import brainbot_core
from brainbot_core.transport import BaseZMQServer

from ..core.brain import CognitiveBrain
from ..core.config import Config
from .payload import brain_action_to_payload, payload_to_observation

logger = logging.getLogger(__name__)


class CuriosityPolicyServer(BaseZMQServer):
    """把七层认知大脑作为 Brainbot AI 提供者的 ZMQ 策略服务器。"""

    def __init__(self, brain: CognitiveBrain, host: str = "127.0.0.1",
                 port: int = 5555, api_token: str | None = None):
        super().__init__(host=host, port=port, api_token=api_token)
        self.brain = brain
        self.register_endpoint("get_action", self._get_action)
        self.register_endpoint("get_state", self._get_state)
        self.register_endpoint("sleep_compress", self._sleep_compress)

    # ------------------------------------------------------------------
    def _get_action(self, data: dict[str, Any]) -> dict[str, Any]:
        """Brainbot AICommandProvider 调用的推理端点。"""
        observation = payload_to_observation(data or {})
        brain_action = self.brain.think(observation)
        return {
            "action": brain_action_to_payload(brain_action),
            "text": brain_action.text,
            "target_id": brain_action.target_id,
        }

    def _get_state(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """查询大脑内部状态（调试/可视化）。"""
        return {
            "memory": self.brain.memory_summary(),
            "last_uncertainty": self.brain._last_uncertainty,  # noqa: SLF001
            "last_mode": self.brain._last_mode.value if self.brain._last_mode else None,  # noqa: SLF001
            "recent_log": self.brain.recent_log(10),
        }

    def _sleep_compress(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """触发睡眠压缩（24h 周期，可手动）。"""
        return self.brain.compressor.maybe_compress(force=True)


def main():
    parser = argparse.ArgumentParser(description="Curiosity AI Policy Server (Brainbot provider)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    brain = CognitiveBrain(config_dir=args.config_dir)
    server = CuriosityPolicyServer(brain, host=args.host, port=args.port)
    print(f"[policy_server] CuriosityPolicyServer 启动于 tcp://{args.host}:{args.port}")
    try:
        server.run()
    except KeyboardInterrupt:
        print("[policy_server] 停止")


if __name__ == "__main__":
    main()
