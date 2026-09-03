"""端到端闭环测试：七层认知架构在轻量仿真中的完整行为。

覆盖验证清单中的核心项:
  - 好奇触发（未知物体）
  - 双路径决策（问人 / 探索）
  - 主动提问 → 人类回答 → 记忆写入 → 终身学习
  - 安全门禁（低电量 → 强制返回充电）
  - Brainbot 适配器（ZMQ 策略服务器握手 + get_action）
  - MuJoCo 物理后端闭环
"""
import time

import numpy as np
import pytest

from src.brainbot_adapters.policy_server import CuriosityPolicyServer
from src.core.brain import CognitiveBrain
from src.core.config import Config
from src.core.types import DecisionPath, SafetyAction
from simulation.world_sim import WorldSim

N_STEPS = 30


def _run_episode(brain, sim, steps=N_STEPS, answer_on=None):
    obs = sim.reset()
    events = []
    for i in range(steps):
        action = brain.think(obs)
        events.append({
            "step": i,
            "mode": brain._last_mode.value,
            "decision": brain._last_decision.path.value if brain._last_decision else None,
            "question": action.text,
            "memory": brain.memory_summary()["total"],
        })
        obs = sim.step(action)
        if answer_on is not None and i in answer_on:
            pass
    return events, obs


def test_closed_loop_ask_path_lifelong_learning():
    """场景1: 人类在场且空闲 → 未知物体 → 提问 → 学习 → 不再重复提问。"""
    brain = CognitiveBrain(Config())
    sim = WorldSim(seed=1, human_present=True, human_idle=True)

    events, _ = _run_episode(brain, sim, steps=N_STEPS)
    decisions = [e["decision"] for e in events]
    questions = [e["question"] for e in events if e["question"] and e["question"].startswith("请问")]

    # 出现过提问路径且生成过问题
    assert DecisionPath.ASK_HUMAN.value in decisions
    assert len(questions) >= 1
    # 学习到了东西（记忆写入）
    assert brain.memory_summary()["total"] >= 1
    # 关键：学习完成后不再重复提问（终身学习收敛）
    learned_at = next(i for i, e in enumerate(events) if e["question"])
    later = [e["question"] for e in events[learned_at + 5:]]
    assert all(q is None or not q.startswith("请问") for q in later)


def test_closed_loop_explore_path_inspect_label():
    """场景2: 人类不在场 + 物体带可见标签 → 探索路径 → inspect 读取标签 → 学习。"""
    brain = CognitiveBrain(Config())
    # 只放一个带标签的未知物体
    sim = WorldSim(seed=2, human_present=False, human_idle=False, objects=[
        {"id": "obj_labelled", "category": "bottle", "color": (0.1, 0.3, 0.9), "size": 0.3,
         "state": "static", "danger": 0.1, "touchable": True, "pos": (1.0, 0.0),
         "label_visible": True},
    ])
    events, _ = _run_episode(brain, sim, steps=40)
    decisions = [e["decision"] for e in events]
    assert DecisionPath.EXPLORE.value in decisions
    assert brain.memory_summary()["total"] >= 1        # 通过探索学到了类别
    mem = brain.store.get_by_object_id("obj_labelled")
    assert mem is not None and mem["object_attrs"]["category"] == "bottle"


def test_safety_gate_recharge():
    """低电量 → 安全门禁强制返回充电（快系统优先）。"""
    brain = CognitiveBrain(Config())
    sim = WorldSim(seed=3, battery_pct=5.0)
    obs = sim.reset()
    action = brain.think(obs)
    assert action.actions["linear"] == 0.0
    assert "safety" in (action.text or "")


def test_brainbot_policy_server_zmq():
    """Brainbot 适配器：策略服务器 ZMQ 握手 + get_action 推理。"""
    import threading
    import zmq

    brain = CognitiveBrain(Config())
    server = CuriosityPolicyServer(brain, host="127.0.0.1", port=5599)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(0.3)

    ctx = zmq.Context()      # 独立 context，避免与 server 共享单例导致 term 阻塞
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://127.0.0.1:5599")
    try:
        # ping
        sock.send(b"\x82\xa8endpoint\xa4ping\xa4data\x80")
        resp = sock.recv()
        assert b"running" in resp or b"ok" in resp

        # get_action
        sim = WorldSim(seed=4, human_present=True, human_idle=True)
        obs = sim.reset()
        from src.brainbot_adapters.payload import payload_to_observation
        # 把 Observation 序列化为载荷
        payload = {
            "proprio": {
                "joint_angles": obs.proprio.joint_angles.tolist(),
                "torques": obs.proprio.torques.tolist(),
                "collision": obs.proprio.collision,
                "battery_pct": obs.proprio.battery_pct,
                "pose": list(obs.proprio.pose),
                "linear_vel": obs.proprio.linear_vel,
                "angular_vel": obs.proprio.angular_vel,
            },
            "scene": {
                "scene_vector": obs.scene.scene_vector.tolist(),
                "objects": [{
                    "object_id": o.object_id, "category": o.category,
                    "color": list(o.color), "size": o.size, "state": o.state,
                    "touchable": o.touchable, "danger_level": o.danger_level,
                    "feature": o.feature.tolist(), "position": list(o.position),
                } for o in obs.scene.objects],
            },
            "human": {"present": True, "idle": True, "answer": None},
            "task": {"urgent": False, "type": "active_learning"},
        }
        import msgpack
        req = {"endpoint": "get_action", "data": payload}
        sock.send(msgpack.packb(req))
        resp = msgpack.unpackb(sock.recv())
        assert "action" in resp
        assert isinstance(resp["action"], dict)
    finally:
        sock.close()
        ctx.term()
        server.running = False


def test_mujoco_bridge_physics():
    """MuJoCo 物理后端：差速运动学让机器人实际移动。"""
    from simulation.mujoco_bridge import MuJoCoBridge
    bridge = MuJoCoBridge()
    bridge.reset()
    obs0 = bridge.get_observation()
    for _ in range(20):
        bridge.step(linear=0.2, angular=0.0, dt=0.05)
    obs1 = bridge.get_observation()
    dist = np.hypot(obs1.proprio.pose[0] - obs0.proprio.pose[0],
                    obs1.proprio.pose[1] - obs0.proprio.pose[1])
    assert dist > 0.05


def test_visualization_ready_log():
    """确保闭环日志可结构化输出（供可视化）。"""
    brain = CognitiveBrain(Config())
    sim = WorldSim(seed=5, human_present=True, human_idle=True)
    events, _ = _run_episode(brain, sim, steps=8)
    assert len(events) == 8
    assert all("mode" in e and "decision" in e for e in events)
