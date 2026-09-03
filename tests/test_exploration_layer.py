"""探索层单元测试：好奇心引擎 / 动作预案 / 双路径决策 / 提问生成。"""
from src.core.config import Config
from src.core.types import (DecisionPath, ExplorationMode, ObjectPercept,
                            UncertaintySignal)
from src.exploration_layer.action_proposer import ActionProposer
from src.exploration_layer.curiosity_engine import CuriosityEngine
from src.exploration_layer.decision_engine import DecisionEngine
from src.exploration_layer.question_generator import QuestionGenerator
from src.understanding_layer.world_model import WorldModel


def _obj(danger=0.1, state="static", category=None, obj_id="obj_1"):
    return ObjectPercept(object_id=obj_id, category=category, color=(0.9, 0.1, 0.1),
                         size=0.5, state=state, danger_level=danger)


def test_curiosity_mode_by_intrinsic():
    ce = CuriosityEngine(Config(), backend="interest")
    assert ce.select_mode(prediction_error=0.7) == ExplorationMode.BROAD_SCAN
    assert ce.select_mode(prediction_error=0.35) == ExplorationMode.DEEP_DIVE
    assert ce.select_mode(prediction_error=0.15) == ExplorationMode.REVIEW
    assert ce.select_mode(prediction_error=0.01) == ExplorationMode.SAFE_IDLE
    u = UncertaintySignal(trigger=True, object_id="x", uncertainty_score=0.9)
    assert ce.select_mode(uncertainty=u) == ExplorationMode.BROAD_SCAN


def test_action_proposer_deep_dive_on_unknown():
    wm = WorldModel(Config(), backend="lightweight")
    ap = ActionProposer(wm, Config())
    z = wm.latent_dim * [0.5]
    import numpy as np
    plan = ap.propose(np.asarray(z, dtype=np.float32), np.zeros(3, dtype=np.float32),
                      _obj(category=None), ExplorationMode.DEEP_DIVE)
    assert plan.action in ("inspect", "approach")


def test_decision_engine_ask_when_human_idle():
    de = DecisionEngine(Config())
    d = de.decide(_obj(danger=0.1),
                  human={"present": True, "idle": True},
                  task={"urgent": False}, exploration_cost=0.5)
    assert d.path == DecisionPath.ASK_HUMAN
    assert d.score >= 6.0


def test_decision_engine_explore_when_human_absent():
    de = DecisionEngine(Config())
    d = de.decide(_obj(danger=0.1),
                  human={"present": False, "idle": False},
                  task={"urgent": False}, exploration_cost=0.5)
    assert d.path == DecisionPath.EXPLORE
    assert d.score < 6.0


def test_decision_engine_urgent_task_drops_ask():
    de = DecisionEngine(Config())
    d = de.decide(_obj(danger=0.1),
                  human={"present": True, "idle": True},
                  task={"urgent": True}, exploration_cost=0.5)
    assert d.path == DecisionPath.EXPLORE


def test_question_generator_rate_limit():
    qg = QuestionGenerator(max_per_minute=3)
    obj = _obj(category=None, obj_id="o")
    qs = [qg.generate(obj, "scene_a") for _ in range(3)]
    assert all(q is not None for q in qs)
    assert qg.generate(obj, "scene_a") is None          # 第 4 次被限流
    # 不同场景不受影响
    assert qg.generate(obj, "scene_b") is not None
    assert "物体" in qs[0]
