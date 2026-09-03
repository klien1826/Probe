"""判断层单元测试：安全门禁 / 认知仲裁 / 记忆检索。"""
import numpy as np

from src.core.config import Config
from src.core.types import (Decision, DecisionPath, ObjectPercept,
                            ProprioState, SafetyAction)
from src.judgment_layer.cognitive_arbiter import CognitiveArbiter
from src.judgment_layer.memory_retriever import MemoryRetriever
from src.judgment_layer.safety_gate import SafetyGate
from src.memory.vector_store import VectorStore


def _proprio(battery=100.0, cop=0.0, drop=0.0, joint=None, vel=0.0):
    return ProprioState(joint_angles=np.zeros(6) if joint is None else joint,
                        torques=np.zeros(6), collision=False, battery_pct=battery,
                        pose=(0, 0, 0), cop_offset=cop, front_drop=drop,
                        linear_vel=vel, angular_vel=0.0)


def test_safety_gate_actions():
    sg = SafetyGate(Config())
    assert sg.check(_proprio()) == SafetyAction.NORMAL
    assert sg.check(_proprio(cop=0.2)) == SafetyAction.HALT
    assert sg.check(_proprio(drop=0.5)) == SafetyAction.WAIT
    assert sg.check(_proprio(battery=5.0)) == SafetyAction.RECHARGE
    assert sg.check(_proprio(joint=np.full(6, 0.99))) == SafetyAction.SOFT_STOP


def test_safety_gate_clamp():
    sg = SafetyGate(Config())
    out = sg.clamp({"linear": 99.0, "angular": -99.0})
    assert out["linear"] <= sg.max_linear
    assert abs(out["angular"]) <= sg.max_angular


def test_arbiter_dangerous_object_forbids_touch():
    arb = CognitiveArbiter(Config())
    d = Decision(path=DecisionPath.EXPLORE, score=3.0)
    obj = ObjectPercept(object_id="o", category="high_voltage", color=(1, 0, 0),
                        size=0.5, danger_level=0.95)
    v = arb.arbitrate(d, obj, {"present": True, "idle": True}, {"urgent": False})
    assert v.allow_touch is False


def test_arbiter_over_questioning_forces_explore():
    arb = CognitiveArbiter(Config())
    d = Decision(path=DecisionPath.ASK_HUMAN, score=7.0)
    obj = ObjectPercept(object_id="o", category=None, color=(1, 0, 0), size=0.5)
    for _ in range(arb.max_questions_per_min):
        arb.record_ask()
    v = arb.arbitrate(d, obj, {"present": True, "idle": True}, {"urgent": False})
    assert v.allow_ask is False
    assert v.force_mode == "broad_scan"


def test_arbiter_retry_limit_marks_unsolvable():
    arb = CognitiveArbiter(Config())
    obj = ObjectPercept(object_id="o", category=None, color=(1, 0, 0), size=0.5)
    for _ in range(arb.max_retry):
        arb.record_retry("o")
    d = Decision(path=DecisionPath.EXPLORE, score=3.0)
    v = arb.arbitrate(d, obj, {"present": True, "idle": True}, {"urgent": False})
    assert v.blocked is True


def test_memory_retriever_known():
    store = VectorStore(dim=512, backend="numpy")
    store.add({"object_id": "obj_a", "scene_vector": np.ones(512),
               "object_attrs": {"category": "cup"},
               "qa_knowledge": {"question": "q", "answer": "a"},
               "safety_rules": {"touchable": True},
               "usage_scenario": {}})
    ret = MemoryRetriever(store)
    known, sim = ret.is_known("obj_a", np.ones(512))
    assert known and sim == 1.0
    hits = ret.retrieve(np.ones(512), k=5)
    assert len(hits) >= 1
