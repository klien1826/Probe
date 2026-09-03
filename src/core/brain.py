"""核心编排 · 七层认知闭环大脑。

把输入层 → 理解层 → 探索层 → 判断层 → 长效记忆层 → Brainbot 适配器
串成一个闭环:
    感知 → 预测误差 → 预测方差触发 → 双路径决策(问人/探索) →
    安全仲裁 → 记忆写入 → 动作输出

外部接口:
    brain.think(observation) -> BrainAction
    （供 Brainbot policy_server / command_provider / 仿真环境调用）
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from ..core.config import Config
from ..core.types import (
    ActionPlan, BrainAction, Decision, DecisionPath, ExplorationMode,
    ObjectPercept, Observation, SafetyAction, ScenePercept,
)
from ..exploration_layer.action_proposer import ActionProposer
from ..exploration_layer.curiosity_engine import CuriosityEngine
from ..exploration_layer.decision_engine import DecisionEngine
from ..exploration_layer.question_generator import QuestionGenerator
from ..input_layer.audio_processor import AudioProcessor
from ..input_layer.proprioception import Proprioception
from ..input_layer.vision_encoder import VisionEncoder
from ..judgment_layer.cognitive_arbiter import CognitiveArbiter
from ..judgment_layer.memory_retriever import MemoryRetriever
from ..judgment_layer.safety_gate import SafetyGate
from ..memory.knowledge_updater import KnowledgeUpdater
from ..memory.memory_compressor import MemoryCompressor
from ..memory.vector_store import VectorStore
from ..understanding_layer.prediction_error import PredictionError
from ..understanding_layer.uncertainty_trigger import UncertaintyTrigger
from ..understanding_layer.world_model import WorldModel, build_latent

# 动作枚举 → Brainbot 兼容的 actions dict
ACTION_CODES = {"move": 0.0, "rotate": 1.0, "approach": 2.0,
                "inspect": 3.0, "ask": 4.0, "wait": 5.0}
MEMORY_DIM = 512


class CognitiveBrain:
    def __init__(self, cfg: Config | None = None, config_dir: str | None = None):
        self.cfg = cfg or Config(config_dir)
        # ---- 各层实例化 ----
        self.vision = VisionEncoder(self.cfg)
        self.audio = AudioProcessor(self.cfg)
        self.proprio = Proprioception(self.cfg)
        self.wm = WorldModel(self.cfg)
        self.pred_error = PredictionError(self.cfg)
        self.trigger = UncertaintyTrigger(self.cfg)
        self.curiosity = CuriosityEngine(self.cfg)
        self.proposer = ActionProposer(self.wm, self.cfg)
        self.decision = DecisionEngine(self.cfg)
        qcfg = self.cfg.curiosity_of("question")
        self.qgen = QuestionGenerator(max_per_minute=qcfg.get("max_per_minute", 3))
        self.safety = SafetyGate(self.cfg)
        self.arbiter = CognitiveArbiter(self.cfg)
        self.store = VectorStore(dim=MEMORY_DIM)
        self.retriever = MemoryRetriever(self.store)
        self.compressor = MemoryCompressor(self.store)
        self.updater = KnowledgeUpdater(self.store)

        # ---- 闭环状态 ----
        self._z_prev: Optional[np.ndarray] = None
        self._a_prev: Optional[np.ndarray] = None
        self._attended_id: Optional[str] = None
        self._attended: Optional[ObjectPercept] = None
        self._pending_question: Optional[str] = None
        self._asked_object_id: Optional[str] = None
        self._last_mode: Optional[ExplorationMode] = None
        self._last_uncertainty: float = 0.0
        self._last_decision: Optional[Decision] = None
        self._last_action: Optional[BrainAction] = None
        self._log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def think(self, observation: Observation) -> BrainAction:
        """一次闭环决策。返回动作（Brainbot 兼容）。"""
        t0 = time.perf_counter()

        # ========== 1. 快系统：物理安全门禁 ==========
        safety_action = self.safety.check(observation.proprio)
        if safety_action != SafetyAction.NORMAL:
            return self._emit_safety(safety_action, observation)

        # ========== 2. 输入层 → 理解层：构建感知 ==========
        scene = observation.scene
        attended = self._select_attended(scene)
        self._attended = attended
        self._attended_id = attended.object_id if attended else None

        proprio_vec = np.asarray([
            *observation.proprio.pose[:2], observation.proprio.linear_vel,
            observation.proprio.angular_vel, observation.proprio.battery_pct / 100.0,
        ], dtype=np.float32)
        obj_feat = attended.feature if attended is not None else np.zeros(16, dtype=np.float32)
        z = build_latent(proprio_vec, obj_feat, dim=self.wm.latent_dim)

        # ========== 3. 学习：人类已回答待处理问题（先学再感知）==========
        # 关键时序：先摄取回答并把物体标记为"已知"，再进行不确定性判断，
        # 避免对"刚学会的物体"在当前步误触发好奇。
        action_vec = self._action_vector()
        if observation.human.get("answer"):
            learned_id = self._ingest_answer(observation.human["answer"], z)
            if learned_id:
                self.wm.seed_known(learned_id, z, action_vec)

        # ========== 4. 世界模型预测 + 预测误差 ==========
        uncertainty = None
        mem_vec = self._memory_vector(scene, attended)
        object_known, _sim = (
            self.retriever.is_known(self._attended_id, mem_vec)
            if self._attended_id else (True, 1.0))
        if self._z_prev is not None:
            z_pred, variance = self.wm.predict(self._z_prev, self._a_prev, self._attended_id)
            pred = self.pred_error.compute(
                z_pred, z, variance, observation.proprio.collision, self._attended_id)
            uncertainty = self.trigger.evaluate(pred, object_known, self._attended_id)
        else:
            uncertainty = self.trigger.evaluate(
                self.pred_error.compute(z, z, 0.0, observation.proprio.collision, self._attended_id),
                object_known=object_known, object_id=self._attended_id)

        self._last_uncertainty = uncertainty.uncertainty_score if uncertainty else 0.0

        # ========== 5. 好奇心模式 + 动作预案 ==========
        familiarity = self.wm.familiarity(self._attended_id) if self._attended_id else 1.0
        mode = self.curiosity.select_mode(uncertainty, self._last_uncertainty, familiarity)
        self._last_mode = mode

        if uncertainty and uncertainty.trigger and self._attended is not None:
            plan = self.proposer.propose(z, action_vec, self._attended, mode)
        else:
            plan = ActionPlan(action="wait", expected_curiosity_gain=0.0, risk=0.0, cost=0.0)

        # ========== 6/7. 双路径决策 + 认知仲裁（仅"好奇触发"时进入）==========
        # 关键语义：物体已学会（不确定性未触发）→ 不再重复提问，回归安全待机。
        question = None
        decision = None
        final_plan = plan

        if uncertainty and uncertainty.trigger and self._attended is not None:
            decision = self.decision.decide(
                self._attended, observation.human, observation.task,
                exploration_cost=plan.cost)
            verdict = self.arbiter.arbitrate(
                decision, self._attended, observation.human, observation.task)
            if decision.path == DecisionPath.ASK_HUMAN and verdict.allow_ask and not verdict.blocked:
                question = self.qgen.generate(self._attended, self._scene_key(observation))
                if question is not None:
                    self.arbiter.record_ask()
                    self._pending_question = question
                    self._asked_object_id = self._attended_id
                    final_plan = ActionPlan(action="ask", expected_curiosity_gain=0.5, risk=0.0, cost=0.1)
            else:
                self._pending_question = None
                if verdict.force_mode:
                    try:
                        final_plan = self.proposer.propose(
                            z, action_vec, self._attended, ExplorationMode(verdict.force_mode))
                    except ValueError:
                        pass
        else:
            self._pending_question = None
        self._last_decision = decision

        # ========== 8. 记忆 / 世界模型提交 ==========
        # 物体已知时提交动力学样本（模型方差下降 → 不再重复触发好奇）
        learned_id = self._maybe_commit_memory(observation, attended)
        if learned_id:
            self.wm.seed_known(learned_id, z, action_vec)
        if self._z_prev is not None:
            self.wm.update(self._z_prev, self._a_prev, z, self._attended_id,
                           commit=object_known and self._attended_id is not None)

        # ========== 9. 输出动作 ==========
        action = self._to_brain_action(final_plan, safety_action, question)
        self._z_prev = z
        self._a_prev = action_vec
        self._last_action = action
        self._log.append({
            "t": time.perf_counter() - t0,
            "mode": mode.value,
            "uncertainty": self._last_uncertainty,
            "decision": decision.path.value if decision else None,
            "question": question,
            "action": final_plan.action,
        })
        return action

    # ------------------------------------------------------------------
    # 内部逻辑
    # ------------------------------------------------------------------
    def _select_attended(self, scene: Optional[ScenePercept]) -> Optional[ObjectPercept]:
        if scene is None or not scene.objects:
            return None
        # 优先关注未知类别物体
        for obj in scene.objects:
            if obj.category is None:
                return obj
        # 其次最近物体
        return min(scene.objects, key=lambda o: np.hypot(*o.position))

    def _action_vector(self) -> np.ndarray:
        if self._last_action is not None:
            a = self._last_action.actions
            return np.asarray([a.get("linear", 0.0), a.get("angular", 0.0),
                               a.get("action", 5.0)], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _memory_vector(self, scene: Optional[ScenePercept],
                       obj: Optional[ObjectPercept]) -> np.ndarray:
        vec = np.zeros(MEMORY_DIM, dtype=np.float32)
        if scene is not None:
            s = scene.scene_vector.reshape(-1)
            vec[: min(len(s), 256)] = s[: min(len(s), 256)]
        if obj is not None:
            f = obj.feature.reshape(-1)
            vec[256: 256 + min(len(f), 32)] = f[: min(len(f), 32)]
        return vec

    def _ingest_answer(self, answer: str, z: Optional[np.ndarray] = None) -> Optional[str]:
        """把人类回答写入记忆，更新物体类别与世界模型。返回写入的 object_id。"""
        oid = self._asked_object_id or self._attended_id
        if oid is None or not answer:
            return None
        category = _parse_category(answer)
        obj = self._attended
        scene_vec = self._memory_vector(None, obj) if obj else np.zeros(MEMORY_DIM)
        self.store.add({
            "object_id": oid,
            "scene_vector": scene_vec,
            "object_attrs": {
                "category": category,
                "color": obj.color if obj else None,
                "size": obj.size if obj else None,
                "state": obj.state if obj else "static",
            },
            "qa_knowledge": {
                "question": self._pending_question or "",
                "answer": answer,
                "timestamp": time.time(),
            },
            "safety_rules": {"touchable": True, "keep_distance": 0.0},
            "usage_scenario": {"task_type": "active_learning", "environment": "sim"},
            "tier": "L1",
            "verify_count": 1,
        })
        self._pending_question = None
        self._asked_object_id = None
        return oid

    def _maybe_commit_memory(self, observation: Observation,
                             attended: Optional[ObjectPercept]) -> Optional[str]:
        """探索学习中确认到类别时，自动写入记忆。返回写入的 object_id。"""
        if attended is None or attended.category is None:
            return None
        if self.retriever.lookup_object(attended.object_id) is not None:
            return None
        self.store.add({
            "object_id": attended.object_id,
            "scene_vector": self._memory_vector(observation.scene, attended),
            "object_attrs": {
                "category": attended.category,
                "color": attended.color,
                "size": attended.size,
                "state": attended.state,
            },
            "qa_knowledge": {"question": "explored", "answer": attended.category, "timestamp": time.time()},
            "safety_rules": {"touchable": attended.touchable, "keep_distance": 0.0},
            "usage_scenario": {"task_type": "active_learning", "environment": "sim"},
            "tier": "L1",
            "verify_count": 1,
        })
        return attended.object_id

    def _to_brain_action(self, plan: ActionPlan, safety_action: SafetyAction,
                         question: Optional[str]) -> BrainAction:
        actions = {
            "action": ACTION_CODES.get(plan.action, 5.0),
            "linear": 0.15 if plan.action in ("approach", "move") else 0.0,
            "angular": 0.4 if plan.action == "rotate" else 0.0,
        }
        if safety_action == SafetyAction.SOFT_STOP:
            actions = {"action": 5.0, "linear": 0.0, "angular": 0.0}
        actions = self.safety.clamp(actions)
        return BrainAction(actions=actions, text=question, decision=self._last_decision,
                           memory_write=bool(question), target_id=self._attended_id)

    def _emit_safety(self, safety_action: SafetyAction, observation: Observation) -> BrainAction:
        actions = {"action": 5.0, "linear": 0.0, "angular": 0.0}
        if safety_action == SafetyAction.HALT:
            actions["action"] = 5.0
        elif safety_action == SafetyAction.RECHARGE:
            actions["action"] = 5.0
        return BrainAction(actions=actions, text=f"safety:{safety_action.value}",
                           target_id=self._attended_id)

    def _scene_key(self, observation: Observation) -> str:
        return "default"

    # ------------------------------------------------------------------
    def recent_log(self, n: int = 20) -> list[dict[str, Any]]:
        return self._log[-n:]

    def memory_summary(self) -> dict[str, Any]:
        return {
            "total": self.store.count(),
            "L1": len(self.store.all(tier="L1")),
            "L2": len(self.store.all(tier="L2")),
            "L3": len(self.store.all(tier="L3")),
        }


def _parse_category(answer: str) -> str:
    """从人类回答中提取类别关键词。"""
    ans = answer.strip().lower()
    known = ["cup", "box", "ball", "bottle", "book", "toy", "cup", "plate",
             "cat", "dog", "plant", "充电器", "杯子", "盒子", "球", "瓶子", "书"]
    for kw in known:
        if kw in ans:
            return kw
    # 去掉"是/这个/的"等
    cleaned = answer.replace("是", "").replace("这个", "").replace("的", "").strip()
    return cleaned if cleaned else answer
